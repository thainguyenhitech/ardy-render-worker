"""Catalog lookup for the backend: sentence -> ARDY text embedding.

Two lookups, two indexes (they cannot be the same index):

* exact    - hash of the normalized sentence -> 8 KB LLM2Vec vector, fetched from the store.
* nearest  - the LLM2Vec vector of a NEW sentence is exactly what we are missing, so the
             similarity search runs in a second, cheap space (MiniLM, 384-d, ~5 ms on CPU).
             Every catalogued sentence therefore stores a MiniLM vector as well.

The index (sentences + hashes + MiniLM vectors) is loaded into RAM at startup: 43 MB for
50k sentences, under 1 GB for a million. The heavy 4096-d vectors stay in the object store and
are fetched per hit (20-50 ms on R2, instant locally), with a small LRU in front.

    svc = CatalogService("file:///path/to/catalog")
    hit = svc.lookup("A person waves with the right hand.")
    hit.vector      # np.float32[4096]  -> send to the ARDY worker
    hit.exact       # False -> hit.sentence is the nearest match; the query went to the miss queue
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .store import ObjectStore, norm_sentence, sentence_hash

log = logging.getLogger("catalog")

INDEX_KEY = "index.npz"
MANIFEST_KEY = "manifest.json"
MISSES_KEY = "misses.jsonl"
VEC_PREFIX = "vec/"
DIM = 4096  # LLM2Vec output width
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model câu tổng quát không phân biệt tốt các cặp đối nghĩa về hướng: đo được "A person walks
# briskly backward away from the camera" xếp "A person walks forward" (0.687) LÊN TRÊN "A person
# walks backward quickly in a circle" (0.489). Với chuyển động thì đi nhầm hướng còn tệ hơn không
# làm gì, nên phạt thẳng các ứng viên chứa từ trái nghĩa với truy vấn.
ANTONYMS = [
    ("forward", "backward"), ("forward", "back"), ("ahead", "backward"),
    ("left", "right"), ("up", "down"), ("upward", "downward"),
    ("raise", "lower"), ("rise", "fall"), ("stand", "sit"), ("stands", "sits"),
    ("open", "close"), ("push", "pull"), ("in", "out"),
]
ANTONYM_PENALTY = 0.35

# XẾP LẠI THEO TỪ KHOÁ (12/09, user "kiểm tra thuật toán search catalog có hiệu quả không"): MiniLM
# là model câu TỔNG QUÁT, gần như mù với trái/phải và bộ phận — đo trên chính kho: "raises the left
# hand high" chọn "raises the RIGHT hand above" (0,937), "turns head right" xếp "turns the head LEFT"
# (0,960) trên "turns the head sharply to the right" (0,930), "points left" ra "points again" (0,855).
# Thước 213 truy vấn trái/phải/hướng/bộ phận (`catalog/test_xep_lai.py`): cosine + phạt trái nghĩa
# đúng 82 %; cộng thêm điểm từ khoá trên TOP-K rồi chọn lại → 99 % (λ = 0,05–0,30 như nhau, lấy 0,10).
# Điểm = cos + λ × (mỗi từ HƯỚNG/BỘ PHẬN của truy vấn: có trong ứng viên +1, ứng viên chứa từ trái
# nghĩa −2, thiếu −1; ứng viên thêm hướng mà truy vấn không nói −0,5) + 0,5 λ × tỉ lệ từ nội dung chung.
# Similarity trả về vẫn là COSINE của câu được chọn (ngưỡng 0,75 phía backend giữ nguyên nghĩa).
# Tắt: CATALOG_XEP_LAI=0.
XEP_LAI = float(os.getenv("CATALOG_XEP_LAI", "0.10"))
XEP_LAI_TOP = int(os.getenv("CATALOG_XEP_LAI_TOP", "40"))
_TU_HUONG = {"up", "upward", "upwards", "down", "downward", "downwards", "forward", "forwards",
             "backward", "backwards", "back", "left", "right", "side", "sideways"}
_TU_BO_PHAN = {"head", "hand", "hands", "arm", "arms", "leg", "legs", "foot", "feet", "shoulder",
               "shoulders", "knee", "knees", "hip", "hips", "chest", "waist", "torso", "finger",
               "fingers", "thumb", "thumbs", "fist", "fists", "elbow", "elbows", "face", "chin",
               "palm", "palms", "wrist", "wrists"}
_TU_DUNG = {"a", "an", "the", "person", "is", "are", "with", "and", "then", "to", "of", "his", "her",
            "their", "them", "it", "at", "on", "in", "as", "if", "while", "again", "once", "twice",
            "slowly", "quickly", "gently", "firmly", "both"} | _TU_HUONG
_TRAI_NGHIA: dict = {}
for _a, _b in ANTONYMS:
    _TRAI_NGHIA.setdefault(_a, set()).add(_b)
    _TRAI_NGHIA.setdefault(_b, set()).add(_a)


def _goc_tu(w: str) -> str:
    """Cắt đuôi thô (raises→rais, waving→wav) — chỉ để so hai câu có cùng động từ, không cần đúng
    ngữ pháp."""
    for suf in ("ing", "es", "s", "ed"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


@dataclass
class Lookup:
    sentence: str  # the catalogued sentence actually used
    vector: Optional[np.ndarray]  # float32[4096], None when the catalog is empty
    exact: bool
    similarity: float = 1.0
    query: str = ""


def diem_xep_lai(sims: np.ndarray, cand_words: Sequence[set], key: str, lam: Optional[float] = None,
                 top: Optional[int] = None) -> tuple[int, np.ndarray]:
    """Điểm XẾP LẠI cho từng ứng viên trong TOP-K: cosine + từ khoá hướng/bộ phận (xem `XEP_LAI`).
    Trả (chỉ số tốt nhất, mảng điểm — ứng viên ngoài top-k giữ nguyên cosine). Dùng chung cho cả kho catalog
    (`CatalogService._xep_lai`) và danh sách ứng viên bên gọi đưa (`nearest_in`)."""
    lam = XEP_LAI if lam is None else float(lam)
    top = XEP_LAI_TOP if top is None else int(top)
    qw = set(re.findall(r"[a-z]+", key))
    khoa = (qw & _TU_HUONG) | (qw & _TU_BO_PHAN)
    qn = {_goc_tu(w) for w in qw if w not in _TU_DUNG}
    diem = np.array(sims, dtype=np.float32).copy()
    ung = (np.argpartition(-sims, min(top, len(sims) - 1))[:top] if len(sims) > top
           else np.arange(len(sims)))
    best, diem_best = int(ung[0]), -9.0
    for j in ung:
        cw = cand_words[j]
        s = float(sims[j])
        for w in khoa:
            if w in cw:
                s += lam
            elif _TRAI_NGHIA.get(w, set()) & cw:
                s -= 2 * lam
            else:
                s -= lam
        for w in (cw & _TU_HUONG) - qw:                   # ứng viên thêm hướng mà truy vấn không nói
            # thêm hướng NGƯỢC với truy vấn nặng hơn thêm hướng vô can — phạt trái nghĩa đã miễn cho câu chứa cả hai
            s -= 1.5 * lam if (_TRAI_NGHIA.get(w, set()) & qw) else 0.5 * lam
        if qn:
            cn = {_goc_tu(w) for w in cw if w not in _TU_DUNG}
            s += 0.5 * lam * len(qn & cn) / len(qn)
        diem[j] = s
        if s > diem_best:
            diem_best, best = s, int(j)
    return best, diem


class MiniLM:
    """all-MiniLM-L6-v2 with mean pooling, on CPU. ~90 MB, ~5 ms per sentence."""

    def __init__(self, model_name: str = MINILM_MODEL, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def __call__(self, texts: Sequence[str], batch: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch):
            chunk = list(texts[i : i + batch])
            enc = self.tok(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state  # [B,T,384]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, dim=-1)  # cosine == dot product
            out.append(vec.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 384), dtype=np.float32)


class CatalogService:
    def __init__(
        self,
        url: str,
        *,
        min_similarity: float = 0.75,  # dưới mức này thì câu được xếp vào hàng chờ mã hóa;
        # 0.55 đủ để không sai nghĩa, nhưng cử chỉ khớp lỏng (0.65 "vẫy tay chào" -> "waves goodbye"),
        # nên với chatbot đặt cao hơn và chấp nhận mã hóa thêm.
        cache_size: int = 512,
        load_minilm: bool = True,
    ):
        self.store = ObjectStore.open(url)
        self.min_similarity = min_similarity
        self.cache_size = cache_size
        self.lock = threading.RLock()
        self.sentences: List[str] = []
        self.hashes: List[str] = []
        self.minilm_vecs = np.zeros((0, 384), dtype=np.float32)
        self._by_hash: dict[str, int] = {}
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._encoder: Optional[MiniLM] = None
        self._cand_words: Optional[List[set]] = None
        self._cand_cache: dict[str, np.ndarray] = {}     # vector ứng viên bên gọi đưa (`nearest_in`), theo câu chuẩn hoá
        self._want_minilm = load_minilm
        self.manifest: dict = {}
        self.stats = {"exact": 0, "nearest": 0, "empty": 0, "fetch": 0, "cache_hit": 0}
        self.load()

    # ---- index -----------------------------------------------------------------------
    @property
    def encoder(self) -> Optional[MiniLM]:
        """Lazily built so a backend that only does exact lookups never pays for it."""
        if self._encoder is None and self._want_minilm:
            try:
                t0 = time.time()
                self._encoder = MiniLM()
                log.info("MiniLM ready in %.1fs", time.time() - t0)
            except Exception as e:  # noqa: BLE001
                log.warning("MiniLM unavailable (%s); nearest-match falls back to string similarity", e)
                self._want_minilm = False
        return self._encoder

    def load(self) -> None:
        raw = self.store.get(INDEX_KEY)
        with self.lock:
            if raw is None:
                self.sentences, self.hashes = [], []
                self.minilm_vecs = np.zeros((0, 384), dtype=np.float32)
            else:
                import io

                z = np.load(io.BytesIO(raw), allow_pickle=False)
                self.sentences = [str(s) for s in z["sentences"]]
                self.hashes = [str(h) for h in z["hashes"]]
                self.minilm_vecs = z["minilm"].astype(np.float32)
            self._by_hash = {h: i for i, h in enumerate(self.hashes)}
            self._cand_words = None
            self.manifest = json.loads(self.store.get_text(MANIFEST_KEY, "{}") or "{}")
        log.info("Catalog %s: %d sentences (%s)", self.store, len(self.sentences), self.manifest.get("encoder", "?"))

    def reload_if_changed(self) -> bool:
        """Cheap poll for other backend instances' writes. Returns True when the index was reloaded."""
        m = json.loads(self.store.get_text(MANIFEST_KEY, "{}") or "{}")
        if m.get("version") != self.manifest.get("version"):
            self.load()
            return True
        return False

    def __len__(self) -> int:
        return len(self.sentences)

    # ---- lookup ----------------------------------------------------------------------
    def lookup(self, sentence: str, *, record_miss: bool = True) -> Lookup:
        key = norm_sentence(sentence)
        h = sentence_hash(key)
        with self.lock:
            idx = self._by_hash.get(h)
        if idx is not None:
            self.stats["exact"] += 1
            return Lookup(self.sentences[idx], self.vector_for(h), True, 1.0, sentence)
        if not self.sentences:
            self.stats["empty"] += 1
            if record_miss:
                self.record_miss(sentence)
            return Lookup(sentence, None, False, 0.0, sentence)
        near_i, sim = self._nearest(key)
        self.stats["nearest"] += 1
        if record_miss:
            self.record_miss(sentence, nearest=self.sentences[near_i], similarity=sim)
        return Lookup(self.sentences[near_i], self.vector_for(self.hashes[near_i]), False, sim, sentence)

    def _antonym_penalty(self, key: str) -> Optional[np.ndarray]:
        """Trừ điểm những câu chứa từ trái nghĩa về hướng so với truy vấn."""
        words = set(re.findall(r"[a-z]+", key))
        opposites = set()
        for a, b in ANTONYMS:
            if a in words:
                opposites.add(b)
            if b in words:
                opposites.add(a)
        if not opposites:
            return None
        if self._cand_words is None or len(self._cand_words) != len(self.sentences):
            self._cand_words = [set(re.findall(r"[a-z]+", norm_sentence(s))) for s in self.sentences]
        pen = np.zeros(len(self.sentences), dtype=np.float32)
        for i, cw in enumerate(self._cand_words):
            if cw & opposites and not (cw & words & {w for pair in ANTONYMS for w in pair}):
                pen[i] = ANTONYM_PENALTY
        return pen

    def _nearest(self, key: str) -> tuple[int, float]:
        enc = self.encoder
        if enc is not None and len(self.minilm_vecs) == len(self.sentences) and len(self.sentences):
            q = enc([key])[0]
            with self.lock:
                cos = self.minilm_vecs @ q  # both are L2-normalized -> dot == cosine
                sims = cos
                pen = self._antonym_penalty(key)
                if pen is not None:
                    sims = sims - pen
                i = self._xep_lai(sims, key) if XEP_LAI > 0 else int(np.argmax(sims))
            return i, float(np.clip(cos[i], 0.0, 1.0))
        import difflib  # fallback: no MiniLM available

        best = difflib.get_close_matches(key, [norm_sentence(s) for s in self.sentences], n=1, cutoff=0.0)
        i = [norm_sentence(s) for s in self.sentences].index(best[0]) if best else 0
        return i, difflib.SequenceMatcher(None, key, norm_sentence(self.sentences[i])).ratio()

    def _xep_lai(self, sims: np.ndarray, key: str, lam: Optional[float] = None,
                 top: Optional[int] = None) -> int:
        """Chọn lại trong TOP-K theo điểm cosine + từ khoá (xem chú thích `XEP_LAI`). Gọi trong lock."""
        if self._cand_words is None or len(self._cand_words) != len(self.sentences):
            self._cand_words = [set(re.findall(r"[a-z]+", norm_sentence(s))) for s in self.sentences]
        best, _diem = diem_xep_lai(sims, self._cand_words, key, lam, top)
        return best

    # ---- tìm trong DANH SÁCH ỨNG VIÊN bên gọi đưa (kho clip render sẵn) ------------------------
    def nearest_in(self, sentence: str, candidates: Sequence[str], top_n: int = 3) -> list[dict]:
        """Câu gần nghĩa nhất trong `candidates` (không phải cả kho) — cùng MiniLM + phạt trái nghĩa + xếp lại
        từ khoá như `lookup`. Vector ứng viên cache theo câu chuẩn hoá nên gọi lặp với cùng danh sách rẻ.
        Trả top_n mục {sentence, similarity (cosine), score (đã xếp lại)} theo score giảm dần.

        VÌ SAO (12/09, user): "LLM không biết catalog có gì … tag đó matching câu trong catalog, khi dùng clip
        offline thì matching clip bằng model search mini". Kho clip render sẵn của 8010 là một catalog con;
        tra kho phải bằng NGHĨA như tra catalog, không phải bằng tên tag.
        """
        key = norm_sentence(sentence)
        cands = [c for c in candidates if isinstance(c, str) and c.strip()]
        if not cands or not key:
            return []
        enc = self.encoder
        if enc is None:
            import difflib
            r = [(c, difflib.SequenceMatcher(None, key, norm_sentence(c)).ratio()) for c in cands]
            r.sort(key=lambda x: -x[1])
            return [{"sentence": c, "similarity": round(float(s), 4), "score": round(float(s), 4)} for c, s in r[:top_n]]
        keys = [norm_sentence(c) for c in cands]
        thieu = [k for k in dict.fromkeys(keys) if k not in self._cand_cache]
        if thieu:
            vecs = enc(thieu)
            for k, v in zip(thieu, vecs):
                self._cand_cache[k] = v
        M = np.stack([self._cand_cache[k] for k in keys])
        q = enc([key])[0]
        cos = M @ q
        sims = cos.copy()
        words = set(re.findall(r"[a-z]+", key))
        cand_words = [set(re.findall(r"[a-z]+", k)) for k in keys]
        opposites = set()
        for a, b in ANTONYMS:
            if a in words:
                opposites.add(b)
            if b in words:
                opposites.add(a)
        if opposites:
            for i, cw in enumerate(cand_words):
                if cw & opposites and not (cw & words & {w for pair in ANTONYMS for w in pair}):
                    sims[i] -= ANTONYM_PENALTY
        _best, diem = diem_xep_lai(sims, cand_words, key, None, max(XEP_LAI_TOP, len(keys)))
        thu_tu = np.argsort(-diem)[:top_n]
        return [{"sentence": cands[i], "similarity": round(float(np.clip(cos[i], 0.0, 1.0)), 4),
                 "score": round(float(diem[i]), 4)} for i in thu_tu]

    def vector_for(self, h: str) -> Optional[np.ndarray]:
        with self.lock:
            v = self._cache.get(h)
            if v is not None:
                self._cache.move_to_end(h)
                self.stats["cache_hit"] += 1
                return v
        raw = self.store.get(f"{VEC_PREFIX}{h}.f16")
        self.stats["fetch"] += 1
        if raw is None:
            log.warning("catalog index lists %s but the vector object is missing", h)
            return None
        v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        if v.shape != (DIM,):
            log.warning("vector %s has shape %s, expected (%d,)", h, v.shape, DIM)
            return None
        with self.lock:
            self._cache[h] = v
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return v

    # ---- misses ----------------------------------------------------------------------
    def record_miss(self, sentence: str, nearest: str = "", similarity: float = 0.0) -> None:
        line = json.dumps(
            {"sentence": sentence.strip(), "nearest": nearest, "similarity": round(similarity, 4), "t": int(time.time())},
            ensure_ascii=False,
        )
        self.store.append(MISSES_KEY, (line + "\n").encode("utf-8"))

    def pending_misses(self, min_similarity: Optional[float] = None) -> List[str]:
        """Distinct sentences worth encoding: not catalogued and not close enough to an existing one."""
        thr = self.min_similarity if min_similarity is None else min_similarity
        raw = self.store.get_text(MISSES_KEY)
        seen, out = set(), []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = rec.get("sentence", "").strip()
            k = norm_sentence(s)
            if not s or k in seen or sentence_hash(k) in self._by_hash:
                continue
            if rec.get("similarity", 0.0) >= thr:
                continue  # a catalogued sentence already covers it
            seen.add(k)
            out.append(s)
        return out

    def clear_misses(self) -> None:
        self.store.delete(MISSES_KEY)

    # ---- writes ----------------------------------------------------------------------
    def add(self, sentences: Sequence[str], vectors: np.ndarray, minilm: Optional[np.ndarray] = None) -> int:
        """Append new sentences. `vectors` is [N,4096]; MiniLM vectors are computed when omitted."""
        if len(sentences) == 0:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(sentences), DIM):
            raise ValueError(f"vectors must be [{len(sentences)},{DIM}], got {vectors.shape}")
        if minilm is None:
            enc = self.encoder
            if enc is None:
                raise RuntimeError("MiniLM is required to add entries (it builds the search index)")
            minilm = enc([norm_sentence(s) for s in sentences])
        with self.lock:
            new_rows = []  # gom rồi nối MỘT lần; nối từng câu là O(n^2), 14k câu sẽ treo
            for i, s in enumerate(sentences):
                h = sentence_hash(s)
                if h in self._by_hash:
                    continue
                self.store.put(f"{VEC_PREFIX}{h}.f16", vectors[i].astype(np.float16).tobytes())
                self._by_hash[h] = len(self.sentences)
                self.sentences.append(s.strip())
                self.hashes.append(h)
                new_rows.append(minilm[i])
            if new_rows:
                self.minilm_vecs = np.concatenate(
                    [self.minilm_vecs, np.asarray(new_rows, dtype=np.float32)], axis=0
                )
                self._cand_words = None
                self.flush()
        return len(new_rows)

    def flush(self) -> None:
        """Write index.npz + manifest.json. Vector objects are written as they are added."""
        import io

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            sentences=np.array(self.sentences, dtype=object).astype("U"),
            hashes=np.array(self.hashes, dtype=object).astype("U"),
            minilm=self.minilm_vecs.astype(np.float16),
        )
        self.store.put(INDEX_KEY, buf.getvalue())
        self.manifest = {
            "count": len(self.sentences),
            "dim": DIM,
            "encoder": self.manifest.get("encoder", "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"),
            "search": MINILM_MODEL,
            "version": int(time.time() * 1000),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.store.put(MANIFEST_KEY, json.dumps(self.manifest, indent=1).encode("utf-8"))
