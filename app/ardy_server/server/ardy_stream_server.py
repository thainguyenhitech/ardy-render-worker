#!/usr/bin/env python3
"""ARDY real-time motion streaming server.

FastAPI app exposing NVIDIA ARDY (autoregressive text-to-motion) as a real-time
skeleton stream for Three.js / VRM / Mixamo clients. Designed for RunPod
Serverless *load-balancing* endpoints (direct HTTP + WebSocket to the worker).

Endpoints
---------
GET  /ping                      RunPod health check: 200 when models are loaded, 204 while loading.
GET  /health                    JSON status (models, sessions, timings).
GET  /skeleton?model=core8      Joint names, parents, T-pose rest positions, axes, fps.
WS   /ws                        Bidirectional streaming (see PROTOCOL below).
POST /sessions                  HTTP fallback: create a session  -> {"session_id": ...}
GET  /sessions/{id}/events      HTTP fallback: Server-Sent Events frame stream.
POST /sessions/{id}/prompt      HTTP fallback: update the prompt (body {"text": "..."}).
POST /sessions/{id}/control     HTTP fallback: {"action": "pause"|"resume"|"reset"|"stop"}.
DELETE /sessions/{id}
POST /generate                  Stateless: one cue -> one bounded clip (queue-endpoint shape).

PROTOCOL (WebSocket, JSON text frames)
--------------------------------------
client -> server
  {"type":"start", "prompt":"A person is walking.", "model":"core8",
   "stream_fps":30, "space":"local"|"global"|"both", "format":"named"|"array",
   "cfg_text":2.0, "cfg_constraint":2.0, "steps":null, "history_frames":4,
   "position":[0,0,0], "heading":0.0, "session_id": "<resume an existing session>", "last_seq": <int>,
   "history":[<motion_frame messages or {"t":..,"root":[..],"quats":[[..],..]}>, ...]}
     - "history": the client's most recent frames (any fps, "t" in seconds); they are resampled to
       the model fps and used as ARDY's initial history so a NEW session continues from the pose
       the client is currently showing instead of the default T-pose (no snap when a stream is
       closed between motions and reopened later). Ignored when "session_id" resumes a live session.
  {"type":"prompt", "text":"A person jumps."}        # takes effect after ~replan_buffer frames
  {"type":"pause"} {"type":"resume"} {"type":"reset"} {"type":"stop"} {"type":"ping"}
server -> client
  {"type":"session", "session_id":..., "model":..., "model_fps":20, "stream_fps":30,
   "joints":[...27 names...], "parents":[-1,0,...], "rest_positions":[[x,y,z],...],
   "units":"m", "up_axis":"Y", "forward_axis":"+Z", "quaternion_order":"xyzw", "space":...}
  {"type":"motion_frame", "seq":104, "t":3.4667, "timestamp":1725612345.678,
   "data":{"bones":{"Hips":[x,y,z,qx,qy,qz,qw], "Spine":[qx,qy,qz,qw], ...}},
   "contacts":[l_heel,l_toe,r_heel,r_toe]}
     - space "global" puts world-space quaternions in "bones"; "both" adds "bones_global".
     - format "array" sends {"root":[x,y,z], "quats":[[qx,qy,qz,qw],...]} in joint order.
  {"type":"status", ...}   {"type":"pong"}   {"type":"error", "message":...}

Rotation convention: ARDY's Core skeleton rest pose is a T-pose with identity joint
rotations (Y up, character facing +Z, meters). "local" quaternions are parent-relative
rotations w.r.t. that rest pose; "global" quaternions are world rotations w.r.t. the same
rest pose. See client/ardy_three_client.js for Mixamo / VRM retargeting.

Environment
-----------
PORT (8000) ARDY_MODEL (core8) ARDY_PRELOAD_MODELS (core8) ARDY_MAX_SESSIONS (4)
ARDY_LEAD_SECONDS (1.0) ARDY_SESSION_IDLE_TIMEOUT (120) ARDY_WARMUP (1)
ARDY_TEXT_EMBEDDINGS  "none" = stateless (embedding comes with each request), or a catalog .npz (build_catalog.py); when set, the 16 GB LLM2Vec
                      text encoder is NOT loaded and prompts are looked up in the catalog instead.
TEXT_ENCODERS_DIR HF_HOME LOCAL_CACHE HF_HUB_OFFLINE  (see prepare_text_encoder.py)

Run:  python ardy_stream_server.py            (serve)
      python ardy_stream_server.py --selftest (load + generate a few windows, print timings)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import base64
import difflib
import math
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

# NOTE: these must be module-level imports. With `from __future__ import annotations` FastAPI
# resolves handler annotations (e.g. `ws: WebSocket`) against module globals; a locally imported
# `WebSocket` would not be found, the parameter would be treated as a required query param, and
# every WebSocket handshake would be rejected with 403 before accept().
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
log = logging.getLogger("ardy-server")

PORT = int(os.getenv("PORT", "8000"))
DEFAULT_MODEL = os.getenv("ARDY_MODEL", "core8")
PRELOAD_MODELS = [m.strip() for m in os.getenv("ARDY_PRELOAD_MODELS", DEFAULT_MODEL).split(",") if m.strip()]
MAX_SESSIONS = int(os.getenv("ARDY_MAX_SESSIONS", "4"))
LEAD_SECONDS = float(os.getenv("ARDY_LEAD_SECONDS", "1.0"))
SESSION_IDLE_TIMEOUT = float(os.getenv("ARDY_SESSION_IDLE_TIMEOUT", "120"))
CLIENT_TIMEOUT = float(os.getenv("ARDY_CLIENT_TIMEOUT", "45"))  # close a WS that sends nothing for this long
WARMUP = os.getenv("ARDY_WARMUP", "1") == "1"
WARMUP_PROMPT = os.getenv("ARDY_WARMUP_PROMPT", "A person is walking.")
MAX_KEEP_FRAMES = 600  # trim session history beyond this many model frames
CROSSFADE_FRAMES = int(os.getenv("ARDY_CROSSFADE_FRAMES", "6"))  # model frames blended old->new after a prompt switch
TEXT_EMBEDDINGS = os.getenv("ARDY_TEXT_EMBEDDINGS", "")  # catalog .npz -> the text encoder is not loaded
ROUND = 5  # decimals in JSON payloads


# --------------------------------------------------------------------------------------
# Math helpers (numpy, xyzw quaternions)
# --------------------------------------------------------------------------------------
def mats_to_quats(mats: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotation matrices -> [..., 4] quaternions (x, y, z, w)."""
    from scipy.spatial.transform import Rotation as R

    shape = mats.shape[:-2]
    q = R.from_matrix(mats.reshape(-1, 3, 3)).as_quat()
    return q.reshape(*shape, 4).astype(np.float32)


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-path slerp between [J,4] xyzw quaternion arrays."""
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    lin = d > 0.9995
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    st = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        w0 = np.where(lin, 1.0 - a, np.sin((1.0 - a) * theta) / st)
        w1 = np.where(lin, a, np.sin(a * theta) / st)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def rnd(x) -> list:
    return [round(float(v), ROUND) for v in x]


def decode_embedding(v) -> Optional[np.ndarray]:
    """Accept a text vector as base64 fp16 (11 KB) or a plain float list."""
    if v is None:
        return None
    if isinstance(v, str):
        arr = np.frombuffer(base64.b64decode(v), dtype=np.float16).astype(np.float32)
    else:
        arr = np.asarray(v, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"embedding must be 1-D, got shape {arr.shape}")
    return arr


def norm_prompt(s: str) -> str:
    """Catalog key: case- and whitespace-insensitive (must match build_catalog.py::norm)."""
    return re.sub(r"\s+", " ", s.strip()).lower()


def quats_to_mats(q: np.ndarray) -> np.ndarray:
    """[...,4] xyzw unit quaternions -> [...,3,3] rotation matrices."""
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


# --------------------------------------------------------------------------------------
# Session / frame state
# --------------------------------------------------------------------------------------
@dataclass
class Frame:
    root_pos: np.ndarray  # [3]
    local_q: np.ndarray  # [J,4] xyzw
    global_q: np.ndarray  # [J,4] xyzw
    contacts: np.ndarray  # [4] bool


@dataclass
class Session:
    id: str
    model_name: str
    prompt: str
    stream_fps: float = 30.0
    space: str = "local"  # local | global | both
    fmt: str = "named"  # named | array
    cfg: Tuple[float, float] = (2.0, 2.0)
    steps: Optional[int] = None
    history_frames: int = 4
    init_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    init_heading: float = 0.0
    postprocess: bool = False
    embedding: Optional[np.ndarray] = None  # text vector supplied by the backend (no encoder, no catalog)

    motion_tensor: Any = None  # torch [1,T,D] normalized (GPU)
    frames: List[Frame] = field(default_factory=list)
    frame_offset: int = 0  # absolute index of frames[0]
    playhead: int = 0  # absolute model-frame index being displayed
    sent_count: int = 0  # stream frames emitted so far
    pending_prompt: Optional[str] = None
    paused: bool = False
    closed: bool = False
    attached: int = 0  # number of live consumers
    last_seen: float = field(default_factory=time.time)
    created: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    gen_count: int = 0
    last_gen_time: float = 0.0
    replan_count: int = 0
    underruns: int = 0
    seeded: int = 0  # model frames of client-provided history this session started from
    burst: bool = False  # BURST: generate+send as fast as possible, no wall-clock pacing
    burst_frames: int = 0  # model frames to produce in burst mode (then status=done)

    def available(self) -> int:
        """Absolute index one past the last generated model frame."""
        return self.frame_offset + len(self.frames)

    def get_frame(self, abs_idx: int) -> Optional[Frame]:
        i = abs_idx - self.frame_offset
        if 0 <= i < len(self.frames):
            return self.frames[i]
        return None


# --------------------------------------------------------------------------------------
# Engine: model loading + generation thread
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        # Chọn thiết bị. ARDY_DEVICE=cpu|mps|cuda để ép tay.
        #
        # Mặc định CỐ Ý bỏ qua MPS dù máy Apple có GPU. Đã đo trên M4, cùng một câu, clip 2,5 giây:
        # CPU 1788 ms, MPS 2357 ms — GPU CHẬM HƠN 32%. Lý do: mô hình chỉ 326M tham số chạy ở lô 3,
        # mỗi bước khử nhiễu là hàng nghìn kernel tí hon, nên thời gian nằm ở chi phí điều phối
        # kernel của Metal chứ không ở phép nhân ma trận. GPU rời chỉ thắng khi kernel đủ lớn.
        #
        # Muốn thử lại MPS thì cần ARDY_DEVICE=mps kèm PYTORCH_ENABLE_MPS_FALLBACK=1, và phần dựng
        # mô hình bên dưới đã xử lý hai chỗ ARDY không tương thích Metal (float64, self.device).
        want = os.getenv("ARDY_DEVICE", "auto").lower()
        if want in ("cpu", "mps", "cuda"):
            self.device = want
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = None
        self.models: Dict[str, Any] = {}
        self.ready = False
        self.error: Optional[str] = None
        self.load_seconds = 0.0
        self.sessions: Dict[str, Session] = {}
        self.sessions_lock = threading.Lock()
        self.wake = threading.Event()
        self.text_cache: "OrderedDict[str, Any]" = OrderedDict()
        self.embeddings: Optional[Dict[str, "np.ndarray"]] = None  # catalog: normalized sentence -> [4096]
        self.embedding_keys: List[str] = []
        self.misses: "OrderedDict[str, int]" = OrderedDict()  # prompts not in the catalog
        # SONG SONG SINH (13/09): khoá MỘT của cả engine làm 8 luồng gọi cùng service chỉ nhanh hơn 1 luồng 25 %
        # trong khi GPU mới ăn 92–121 W trên 230 W. Nhiều TIẾN TRÌNH service KHÔNG cứu được: không bật MPS thì các
        # context CUDA chia lượt chứ không chạy chồng (đo 12/09: 6 tiến trình 3,75 s/clip, TỆ hơn 1 tiến trình 3,2).
        # Các LUỒNG trong CÙNG tiến trình dùng chung một context nên kernel chồng được → nới khoá thành SEMAPHORE.
        # `ARDY_GEN_SONG_SONG=1` là hành vi cũ (tuần tự tuyệt đối); mỗi phiên có tensor riêng nên forward song song
        # an toàn, chỉ tốn thêm VRAM theo số luồng.
        self.gen_lock = threading.Semaphore(max(1, int(os.getenv("ARDY_GEN_SONG_SONG", "1"))))
        self.stats = {"windows": 0, "gen_seconds": 0.0, "encode_calls": 0}

    # ---- loading -------------------------------------------------------------------
    def load(self) -> None:
        t0 = time.time()
        try:
            if TEXT_EMBEDDINGS.lower() in ("none", "off", "0"):
                # Fully stateless worker: every request carries its own text vector. Nothing to load.
                log.info("Text source: none (each request must supply an embedding)")
                self.text_encoder = False
            elif TEXT_EMBEDDINGS:
                self.load_catalog(TEXT_EMBEDDINGS)
                # ARDY accepts text_encoder=False and leaves model.text_encoder unset; every prompt
                # is served from the catalog, so the 16 GB Llama-3-8B encoder is never loaded.
                self.text_encoder = False
            else:
                from ardy.model.load_model import load_text_encoder

                log.info("Loading text encoder on %s ...", self.device)
                self.text_encoder = load_text_encoder(mode="local", device=self.device)
            for name in PRELOAD_MODELS:
                self.get_model(name)
            if WARMUP:
                self._warmup()
            self.load_seconds = time.time() - t0
            self.ready = True
            log.info("Engine ready in %.1fs (models=%s)", self.load_seconds, list(self.models))
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log.exception("Engine failed to load")

    def get_model(self, name: str):
        if name not in self.models:
            from ardy.model.load_model import load_model

            t0 = time.time()
            log.info("Loading ARDY model '%s' ...", name)
            if self.device == "mps":
                # Metal không có float64, mà ARDY dựng lịch nhiễu của diffusion bằng float64 nên
                # dựng thẳng trên MPS là hỏng. Dựng trên CPU, hạ mọi float64 xuống float32 rồi mới
                # chuyển sang GPU. Lịch nhiễu ở float32 là chuẩn của phần lớn bản cài diffusion.
                model = load_model(name, device="cpu", text_encoder=self.text_encoder)
                for mod in model.modules():
                    for attr, buf in list(mod.named_buffers(recurse=False)):
                        if buf is not None and buf.dtype == self.torch.float64:
                            setattr(mod, attr, buf.float())
                model = model.to("mps")
                # ARDY lưu `self.device` thành thuộc tính thường lúc dựng (ardy_model.py:74) và
                # dùng nó để tạo tensor tạm trong lúc sinh. `.to()` chỉ chuyển trọng số, không
                # sửa thuộc tính đó, nên nếu quên dòng này thì trọng số ở GPU còn tensor tạm ở CPU.
                model.device = "mps"
            else:
                model = load_model(name, device=self.device, text_encoder=self.text_encoder)
            model.eval()
            self.models[name] = model
            log.info(
                "Model '%s' loaded in %.1fs: fps=%s horizon=%s patch=%s steps=%s joints=%s",
                name,
                time.time() - t0,
                model.motion_rep.fps,
                model.gen_horizon_len,
                model.num_frames_per_token,
                model.diffusion.num_base_steps,
                model.skeleton.nbjoints,
            )
        return self.models[name]

    def _warmup(self) -> None:
        log.info("Warmup ...")
        s = Session(id="warmup", model_name=PRELOAD_MODELS[0], prompt=WARMUP_PROMPT)
        for _ in range(2):
            self.generate_step(s, replan=False)
        log.info("Warmup done (last window %.3fs)", s.last_gen_time)

    # ---- text -----------------------------------------------------------------------
    def load_catalog(self, path: str) -> None:
        """Load sentence -> embedding pairs built by build_catalog.py."""
        z = np.load(path, allow_pickle=False)
        sentences = [str(x) for x in z["sentences"]]
        emb = z["embeddings"].astype(np.float32)
        if len(sentences) != emb.shape[0]:
            raise ValueError(f"catalog mismatch: {len(sentences)} sentences vs {emb.shape[0]} embeddings")
        self.embeddings = {norm_prompt(s): emb[i] for i, s in enumerate(sentences)}
        self.embedding_keys = list(self.embeddings)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        log.info(
            "Text catalog %s: %d sentences, dim %d, %.2f MB (encoder %s)",
            path, len(sentences), emb.shape[1], emb.nbytes / 1e6, meta.get("encoder", "?"),
        )

    def catalog_feat(self, prompt: str):
        """[1,1,D] features + [1,1] mask for a catalogued sentence (nearest match on a miss)."""
        torch = self.torch
        key = norm_prompt(prompt)
        vec = self.embeddings.get(key)
        if vec is None:
            near = difflib.get_close_matches(key, self.embedding_keys, n=1, cutoff=0.0)
            if not near:
                raise RuntimeError("text catalog is empty")
            self.misses[prompt.strip()] = self.misses.get(prompt.strip(), 0) + 1
            while len(self.misses) > 200:
                self.misses.popitem(last=False)
            log.warning("Prompt not in catalog: %r -> using %r", prompt[:70], near[0][:70])
            vec = self.embeddings[near[0]]
        feat = torch.as_tensor(vec, dtype=torch.float32, device=self.device)[None, None]
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        return feat, mask

    def feat_from_vector(self, vec: np.ndarray):
        """[1,1,D] features + [1,1] mask from a caller-supplied text vector."""
        torch = self.torch
        feat = torch.as_tensor(np.ascontiguousarray(vec), dtype=torch.float32, device=self.device)[None, None]
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        return feat, mask

    def encode_text(self, model, prompt: str, embedding: Optional[np.ndarray] = None):
        if embedding is not None:
            return self.feat_from_vector(embedding)
        key = prompt.strip()
        if key in self.text_cache:
            self.text_cache.move_to_end(key)
            return self.text_cache[key]
        t0 = time.time()
        if self.embeddings is not None:
            feat, mask = self.catalog_feat(key)
        elif self.text_encoder is False:
            raise RuntimeError(
                f"no text source on this worker: request must supply an 'embedding' (prompt {prompt[:60]!r})"
            )
        else:
            with self.torch.no_grad():
                feat, mask = model._encode_text([key])
        self.stats["encode_calls"] += 1
        self.text_cache[key] = (feat, mask)
        if len(self.text_cache) > 128:
            self.text_cache.popitem(last=False)
        log.info("Encoded prompt %r in %.3fs", key[:60], time.time() - t0)
        return feat, mask

    # ---- skeleton info --------------------------------------------------------------
    def skeleton_info(self, model_name: str) -> dict:
        model = self.get_model(model_name)
        sk = model.skeleton
        rest = sk.neutral_joints.detach().cpu().numpy()
        rest = rest - rest[sk.root_idx]
        return {
            "model": model_name,
            "model_fps": float(model.motion_rep.fps),
            "gen_horizon_frames": int(model.gen_horizon_len),
            "frames_per_token": int(model.num_frames_per_token),
            "denoising_steps": int(model.diffusion.num_base_steps),
            "joints": list(sk.bone_order_names),
            "parents": [int(p) for p in sk.joint_parents.tolist()],
            "rest_positions": [rnd(r) for r in rest],
            "units": "m",
            "up_axis": "Y",
            "forward_axis": "+Z",
            "quaternion_order": "xyzw",
            "rest_pose": "T-pose with identity joint rotations",
            "contacts": ["left_heel", "left_toe", "right_heel", "right_toe"],
        }

    # ---- sessions -------------------------------------------------------------------
    def create_session(self, cfg: dict) -> Session:
        with self.sessions_lock:
            live = [s for s in self.sessions.values() if not s.closed]
            if len(live) >= MAX_SESSIONS:
                raise RuntimeError(f"max sessions reached ({MAX_SESSIONS})")
        model_name = str(cfg.get("model") or DEFAULT_MODEL)
        model = self.get_model(model_name)
        patch = int(model.num_frames_per_token)
        max_window = (10 * int(model.motion_rep.fps) // patch) * patch
        hist = int(cfg.get("history_frames") or 4)
        hist = max(patch, min(hist, max_window - int(model.gen_horizon_len)))
        hist = (hist // patch) * patch
        steps = cfg.get("steps")
        steps = int(steps) if steps else int(model.diffusion.num_base_steps)
        steps = max(1, min(steps, int(model.diffusion.num_base_steps)))
        s = Session(
            id=str(cfg.get("session_id") or uuid.uuid4().hex[:12]),
            model_name=model_name,
            prompt=str(cfg.get("prompt") or WARMUP_PROMPT),
            stream_fps=float(cfg.get("stream_fps") or 30.0),
            space=str(cfg.get("space") or "local"),
            fmt=str(cfg.get("format") or "named"),
            cfg=(float(cfg.get("cfg_text", 2.0)), float(cfg.get("cfg_constraint", 2.0))),
            steps=steps,
            history_frames=hist,
            init_pos=np.asarray(cfg.get("position") or [0.0, 0.0, 0.0], dtype=np.float32),
            init_heading=float(cfg.get("heading") or 0.0),
            postprocess=bool(cfg.get("postprocess", False)),
            embedding=decode_embedding(cfg.get("embedding")),
        )
        s.burst = bool(cfg.get("burst", False))
        _dur = float(cfg.get("duration_s") or 0.0)
        s.burst_frames = int(round(_dur * float(model.motion_rep.fps))) if (s.burst and _dur > 0) else 0
        if s.space not in ("local", "global", "both"):
            s.space = "local"
        if s.fmt not in ("named", "array"):
            s.fmt = "named"
        s.stream_fps = max(1.0, min(s.stream_fps, 120.0))
        history = cfg.get("history")
        if history:
            try:
                self.seed_session(s, model, history, float(cfg.get("history_fps") or s.stream_fps))
            except Exception as e:  # noqa: BLE001
                log.warning("Session %s: could not seed from client history (%s); starting from rest", s.id, e)
        with self.sessions_lock:
            self.sessions[s.id] = s
        self.wake.set()
        log.info("Session %s created: model=%s fps=%s space=%s hist=%s steps=%s seeded=%d prompt=%r", s.id, model_name, s.stream_fps, s.space, hist, steps, s.seeded, s.prompt)
        return s

    def seed_session(self, s: Session, model, history: list, src_fps: float) -> None:
        """Start a session from the client's recent frames instead of the rest pose.

        The frames (local xyzw quaternions relative to the T-pose, root position in meters — exactly
        what this server streams) are resampled to the model fps, converted back into ARDY features
        with the motion representation's forward transform and installed as the session history, so
        the first generated window continues the pose the client is showing right now.
        """
        torch = self.torch
        joints = list(model.skeleton.bone_order_names)
        mfps = float(model.motion_rep.fps)
        patch = int(model.num_frames_per_token)
        roots, quats, times = [], [], []
        for k, fr in enumerate(history):
            d = fr.get("data", fr) if isinstance(fr, dict) else None
            if not d:
                continue
            if "bones" in d:
                b = d["bones"]
                hip = b.get(joints[0])
                if hip is None or len(hip) < 7:
                    continue
                roots.append(hip[:3])
                quats.append([(b.get(n) or [0, 0, 0, 1])[-4:] for n in joints])
            elif "quats" in d and "root" in d:
                roots.append(d["root"][:3])
                quats.append([q[-4:] for q in d["quats"]])
            else:
                continue
            times.append(float(fr.get("t", k / max(src_fps, 1.0))))
        if not roots:
            raise ValueError("no usable frames")
        roots_a = np.asarray(roots, dtype=np.float32)
        quats_a = np.asarray(quats, dtype=np.float32)
        if quats_a.shape[1] != len(joints):
            raise ValueError(f"expected {len(joints)} joints, got {quats_a.shape[1]}")
        times_a = np.asarray(times, dtype=np.float64)
        # resample (nearest) onto the model fps grid that ends at the newest frame
        span = max(0.0, float(times_a[-1] - times_a[0]))
        n_out = int(round(span * mfps)) + 1
        n_out = max(patch, min(s.history_frames, (n_out // patch) * patch))
        grid = times_a[-1] - np.arange(n_out)[::-1] / mfps
        idx = np.abs(times_a[None, :] - grid[:, None]).argmin(axis=1)
        mats = quats_to_mats(quats_a[idx])  # [T,J,3,3]
        root = roots_a[idx]  # [T,3]
        lr = torch.as_tensor(mats, device=self.device)[None]
        rp = torch.as_tensor(root, device=self.device)[None]
        with torch.no_grad():
            feats = model.motion_rep(lr, rp, to_normalize=True)  # [1,T,D] normalized
            out = model.motion_rep.inverse(model.motion_rep.unnormalize(feats), is_normalized=False)
        frames = self._to_frames(out)
        with s.lock:
            s.motion_tensor = feats
            s.frames = frames
            s.frame_offset = 0
            s.playhead = len(frames) - 1
            # stream from the newest seeded frame (= the pose the client shows), not from the past
            s.sent_count = int(math.ceil((len(frames) - 1) * s.stream_fps / mfps))
            s.seeded = len(frames)
        log.info("Session %s seeded from %d client frames -> %d model frames (span %.2fs)", s.id, len(roots), len(frames), span)

    def get_session(self, sid: str) -> Optional[Session]:
        with self.sessions_lock:
            s = self.sessions.get(sid)
        if s is None or s.closed:
            return None
        return s

    def close_session(self, sid: str) -> None:
        with self.sessions_lock:
            s = self.sessions.pop(sid, None)
        if s is not None:
            s.closed = True
            with s.lock:
                s.motion_tensor = None
                s.frames.clear()
            log.info("Session %s closed (windows=%d replans=%d)", sid, s.gen_count, s.replan_count)

    def reset_session(self, s: Session) -> None:
        with s.lock:
            s.motion_tensor = None
            s.frames.clear()
            s.frame_offset = 0
            s.playhead = 0
            s.sent_count = 0
        self.wake.set()

    # ---- generation -----------------------------------------------------------------
    def generate_step(self, s: Session, replan: bool) -> None:
        """One autoregressive window: append (continuation) or regenerate from the playhead (replan).

        Mirrors scripts/interactive_demo/generation.py::_generate_step for a single sample.
        """
        torch = self.torch
        model = self.get_model(s.model_name)
        patch = int(model.num_frames_per_token)
        horizon = int(model.gen_horizon_len)
        fps = float(model.motion_rep.fps)

        with s.lock:
            mt = s.motion_tensor
            cur_len = 0 if mt is None else int(mt.shape[1])
            playhead_local = s.playhead - s.frame_offset

        if replan:
            # keep enough already-played frames to cover the generation latency
            buffer_frames = max(2, int(math.ceil(max(s.last_gen_time, 0.05) * fps)) + 1)
            history_end_idx = min(cur_len - 1, playhead_local + buffer_frames)
        else:
            history_end_idx = cur_len - 1
        if cur_len >= patch:
            history_end_idx = max(history_end_idx, patch - 1)
        if history_end_idx >= 0:
            history_length = (min(history_end_idx + 1, s.history_frames) // patch) * patch
        else:
            history_length = 0
        history_start_idx = max(0, history_end_idx - history_length + 1)

        hist = None
        if mt is not None and history_length > 0 and history_start_idx <= history_end_idx:
            hist = mt[:, history_start_idx : history_end_idx + 1]
        num_frames = history_length + horizon

        feat, mask = self.encode_text(model, s.prompt, s.embedding)
        init_pos = init_heading = None
        if hist is None:
            init_pos = torch.as_tensor(s.init_pos, dtype=torch.float32, device=self.device)[None]
            init_heading = torch.tensor([s.init_heading], dtype=torch.float32, device=self.device)

        t0 = time.time()
        with self.gen_lock, torch.no_grad():
            samples = model.autoregressive_step(
                num_frames=num_frames,
                num_denoising_steps=int(s.steps or model.diffusion.num_base_steps),
                motion_mask=None,
                observed_motion=None,
                cfg_weight=s.cfg,
                texts=None,
                text_feat=feat,
                text_pad_mask=mask,
                init_history_sequence=hist,
                init_global_translation=init_pos,
                init_first_heading_angle=init_heading,
            )
            new = samples[:, history_length:]  # [1, horizon, D] normalized
            out = model.motion_rep.inverse(model.motion_rep.unnormalize(new), is_normalized=False)
            if s.postprocess:
                try:
                    from ardy.postprocess import post_process_motion

                    corr = post_process_motion(
                        out["local_rot_mats"], out["root_positions"], out["foot_contacts"].float(), model.skeleton
                    )
                    out.update(corr)
                except Exception as e:  # noqa: BLE001
                    log.warning("postprocess failed, disabling for session %s: %s", s.id, e)
                    s.postprocess = False
            frames = self._to_frames(out)
        gen_time = time.time() - t0

        with s.lock:
            if s.closed:
                return
            if mt is None or s.motion_tensor is None:
                s.motion_tensor = new.clone()
                s.frames = frames
                s.frame_offset = 0
            else:
                keep = history_end_idx + 1
                if replan:
                    # Crossfade the already-generated future (old prompt) into the regenerated frames so a
                    # prompt switch never produces a visible snap. Only the emitted poses are blended; the
                    # feature tensor keeps the pure new motion for conditioning.
                    old_future = s.frames[keep : keep + CROSSFADE_FRAMES]
                    for i, f_old in enumerate(old_future):
                        if i >= len(frames):
                            break
                        w = (i + 1) / (len(old_future) + 1)  # weight of the NEW motion, ramps up
                        f_new = frames[i]
                        frames[i] = Frame(
                            ((1.0 - w) * f_old.root_pos + w * f_new.root_pos).astype(np.float32),
                            slerp(f_old.local_q, f_new.local_q, w),
                            slerp(f_old.global_q, f_new.global_q, w),
                            f_new.contacts,
                        )
                s.motion_tensor = torch.cat([s.motion_tensor[:, :keep], new], dim=1)
                del s.frames[keep:]
                s.frames.extend(frames)
            # trim old history
            extra = len(s.frames) - MAX_KEEP_FRAMES
            if extra > 0:
                drop = min(extra, max(0, playhead_local - s.history_frames - 8))
                if drop > 0:
                    s.frames = s.frames[drop:]
                    s.motion_tensor = s.motion_tensor[:, drop:]
                    s.frame_offset += drop
            s.gen_count += 1
            s.last_gen_time = gen_time
            if replan:
                s.replan_count += 1
        self.stats["windows"] += 1
        self.stats["gen_seconds"] += gen_time
        if s.gen_count <= 3 or s.gen_count % 50 == 0 or replan:
            log.info(
                "Session %s window #%d %s: hist=%d frames, %.3fs (%.1fx realtime)",
                s.id,
                s.gen_count,
                "REPLAN" if replan else "cont",
                history_length,
                gen_time,
                (horizon / fps) / max(gen_time, 1e-6),
            )

    def generate_clip(self, cfg: dict) -> dict:
        """Generate a bounded motion clip and return it in one piece (no session, no streaming).

        This is the shape a RunPod *queue* endpoint wants: one job in, one clip out, worker idle
        in between. `history` continues from the pose the client is showing; `embedding` carries the
        text vector so no encoder or catalog is needed on the worker.
        """
        model = self.get_model(str(cfg.get("model") or DEFAULT_MODEL))
        mfps = float(model.motion_rep.fps)
        out_fps = float(cfg.get("fps") or mfps)
        duration = max(0.1, min(float(cfg.get("duration_s") or 3.0), 30.0))
        s = Session(
            id="job-" + uuid.uuid4().hex[:8],
            model_name=str(cfg.get("model") or DEFAULT_MODEL),
            prompt=str(cfg.get("prompt") or WARMUP_PROMPT),
            stream_fps=out_fps,
            space=str(cfg.get("space") or "local"),
            fmt=str(cfg.get("format") or "array"),
            steps=max(1, min(int(cfg.get("steps") or model.diffusion.num_base_steps),
                             int(model.diffusion.num_base_steps))),
            cfg=(float(cfg.get("cfg_text", 2.0)), float(cfg.get("cfg_constraint", 2.0))),
            history_frames=int(cfg.get("history_frames") or 16),
            postprocess=bool(cfg.get("postprocess", False)),
            embedding=decode_embedding(cfg.get("embedding")),
        )
        patch = int(model.num_frames_per_token)
        s.history_frames = max(patch, (s.history_frames // patch) * patch)
        t0 = time.time()
        if cfg.get("target_pose"):
            # NEO HAI ĐẦU: sinh cả clip trong MỘT lượt lấy mẫu để ràng buộc khung cuối có hiệu
            # lực (đường tăng dần từng cửa sổ không nhìn thấy khung cuối). Xem `sinh_neo`.
            seeded = self.sinh_neo(s, model, cfg, duration)
        else:
            seeded = 0
            if cfg.get("history"):
                self.seed_session(s, model, cfg["history"], float(cfg.get("history_fps") or out_fps))
                seeded = len(s.frames)
            need = seeded + int(math.ceil(duration * mfps))
            while s.available() < need:
                self.generate_step(s, replan=False)
        gen_ms = round((time.time() - t0) * 1000, 1)

        joints = list(model.skeleton.bone_order_names)
        # Đo độ trung thành của seed: lệch giữa khung CUỐI của lịch sử đưa vào và khung ĐẦU model
        # sinh ra. Đây là con số cho biết ARDY có nối đúng vào tư thế được giao hay không.
        def _delta(qa, qb) -> float:
            d = np.abs(np.sum(qa * qb, axis=-1)).clip(0.0, 1.0)
            return float(np.degrees(2.0 * np.arccos(d)).max())

        seed_jump = seam_vel = hist_vel = None
        if seeded >= 2 and len(s.frames) > seeded:
            # Lệch qua điểm nối so với lệch giữa hai khung cuối của lịch sử. Nếu hai số xấp xỉ
            # nhau thì chuyển động LIÊN TỤC về vận tốc, và phần "lệch" chỉ là model đang chạy tiếp
            # chứ không phải nhảy tư thế. So sánh vị trí đơn thuần sẽ hiểu nhầm chỗ này.
            seed_jump = round(_delta(s.frames[seeded - 1].local_q, s.frames[seeded].local_q), 2)
            hist_vel = round(_delta(s.frames[seeded - 2].local_q, s.frames[seeded - 1].local_q), 2)
            seam_vel = seed_jump
        n_out = int(round(duration * out_fps))
        frames = []
        for k in range(n_out):
            pos = seeded + k * mfps / out_fps
            i = int(math.floor(pos))
            a = pos - i
            f0 = s.get_frame(i)
            f1 = s.get_frame(i + 1) if a > 1e-6 else None
            if f0 is None:
                break
            msg = build_frame_msg(s, joints, k, f0, f1, a if f1 is not None else 0.0)
            frames.append({"t": round(k / out_fps, 4), "data": msg["data"], "contacts": msg["contacts"]})
        # a clip is played relative to where the character already is: hand the caller the root
        # offset so it can re-anchor instead of teleporting to ARDY's world origin
        root0 = frames[0]["data"]["root"] if frames and "root" in frames[0]["data"] else None
        if root0 is None and frames:
            root0 = frames[0]["data"]["bones"][joints[0]][:3]
        return {
            "model": s.model_name,
            "prompt": s.prompt,
            "fps": out_fps,
            "space": s.space,
            "format": s.fmt,
            "joints": joints,
            "seeded_frames": seeded,
            "seed_jump_deg": seed_jump,
            "hist_vel_deg": hist_vel,   # lệch mỗi khung ngay TRƯỚC điểm nối
            "seam_vel_deg": seam_vel,   # lệch mỗi khung QUA điểm nối
            "root_origin": root0,
            "frames": frames,
            "gen_ms": gen_ms,
            "windows": s.gen_count,
        }

    # ---- neo hai đầu ---------------------------------------------------------------
    def rang_buoc_cuoi(self, model, target_pose: dict, idx_cuoi: int, N: int):
        """(observed_motion, motion_mask) ghim MỘT khung vào tư thế đích.

        `target_pose` = {"quats": [[x,y,z,w] × J] local so T-pose, "root": [x,y,z]} — đúng định
        dạng server này vẫn phát ra, nên bên gọi chỉ việc gửi lại một khung nó đang hiển thị.

        Ràng buộc KHÔNG phải là ghi đè đặc trưng: ARDY có bộ ràng buộc thưa riêng
        (`ardy.constraints`), đổi sang cặp (observed, mask) bằng
        `motion_rep.create_conditions_from_constraints_batched` — chỉ điền đúng các lát vị trí
        khớp toàn cục · root_2d · root_y · hướng thân. Tự đắp mask lên toàn bộ chiều đặc trưng
        thì gần như không ăn (đo 09/09: 67,7° → 56,5°, tức không dẫn được gì).
        """
        from ardy.constraints import FullBodyConstraintSet  # noqa: PLC0415

        torch = self.torch
        joints = list(model.skeleton.bone_order_names)
        q = np.asarray([list(x)[-4:] for x in target_pose["quats"]], dtype=np.float32)
        if q.shape[0] != len(joints):
            raise ValueError(f"target_pose cần {len(joints)} khớp, nhận {q.shape[0]}")
        root = np.asarray(target_pose.get("root") or [0.0, 0.0, 0.0], dtype=np.float32)
        lr = torch.as_tensor(quats_to_mats(q)[None], device=self.device)     # [1,J,3,3]
        rp = torch.as_tensor(root[None], device=self.device)                 # [1,3]
        with torch.no_grad():
            grot, pj, _ = model.skeleton.fk(lr, rp)
            # frame_indices ở CPU: `ardy.constraints.create_pairs` stack nó với `torch.arange(nbjoints)` (CPU) — đặt
            # lên CUDA là "Expected all tensors to be on the same device" (12/09, lần đầu chạy service trên GPU rời;
            # ở nhà device=cpu nên chưa lộ). Dữ liệu (pj/grot) vẫn ở `self.device`.
            cs = FullBodyConstraintSet(
                skeleton=model.skeleton,
                frame_indices=torch.tensor([int(idx_cuoi)]),
                global_joints_positions=pj,
                global_joints_rots=grot,
            )
            return model.motion_rep.create_conditions_from_constraints_batched(
                [cs], torch.tensor([int(N)]), True, self.device)

    def sinh_neo(self, s: Session, model, cfg: dict, duration: float) -> int:
        """Sinh trọn clip trong MỘT lượt lấy mẫu, có thể ghim cả hai đầu. Trả số khung lịch sử.

        Vì sao một lượt chứ không tăng dần từng cửa sổ như `generate_step`: ràng buộc khung CUỐI
        chỉ có nghĩa khi bộ khử nhiễu nhìn thấy cả chuỗi (kênh "future constraints" của nó).

        Đầu clip ghim bằng `history` (điều kiện CỨNG — đo 09/09 lệch 1,64°, gần như tuyệt đối),
        cuối clip ghim bằng ràng buộc thưa (hướng dẫn MỀM — lệch 1,7–4,3°, phần dư để ống hiệu
        chỉnh nuốt nốt bằng tầng tắt độ lệch bậc 5 sẵn có).

        BẪY: `cfg_weight` truyền SỐ VÔ HƯỚNG thì thư viện tự đặt trọng số kênh ràng buộc = 0
        (`ardy_model.py` dòng 275) — ghim vẫn chạy, không lỗi, không cảnh báo, chỉ là không dẫn.
        Phải truyền CẶP (cfg_text, cfg_constraint); `s.cfg` vốn đã là cặp.
        """
        torch = self.torch
        mfps = float(model.motion_rep.fps)
        horizon = int(model.gen_horizon_len)
        hist = None
        seeded = 0
        if cfg.get("history"):
            self.seed_session(s, model, cfg["history"], float(cfg.get("history_fps") or s.stream_fps))
            hist = s.motion_tensor
            seeded = int(hist.shape[1])
        n_model = int(math.ceil(duration * mfps))
        idx_cuoi = seeded + n_model - 1
        # bội của horizon ⇒ không sinh thừa khung sau mốc ghim
        N = seeded + int(math.ceil(n_model / horizon)) * horizon
        obs = mask = None
        tp = cfg.get("target_pose")
        if tp:
            obs, mask = self.rang_buoc_cuoi(model, tp, idx_cuoi, N)
        feat, tmask = self.encode_text(model, s.prompt, s.embedding)
        with self.gen_lock, torch.no_grad():
            samples = model(
                texts=None,
                num_frames=N,
                num_denoising_steps=int(s.steps or model.diffusion.num_base_steps),
                pad_mask=torch.ones(1, N, dtype=torch.bool, device=self.device),
                # khi có lịch sử thì hướng ban đầu do CHÍNH lịch sử quyết (thư viện bắt None)
                first_heading_angle=None if hist is not None else torch.zeros(1, device=self.device),
                motion_mask=mask,
                observed_motion=obs,
                cfg_weight=s.cfg,
                text_feat=feat,
                text_pad_mask=tmask,
                init_history_sequence=hist,
                progress_bar=lambda x, **k: x,
            )
            out = model.motion_rep.inverse(model.motion_rep.unnormalize(samples), is_normalized=False)
            if s.postprocess:
                try:
                    from ardy.postprocess import post_process_motion  # noqa: PLC0415

                    out.update(post_process_motion(out["local_rot_mats"], out["root_positions"],
                                                   out["foot_contacts"].float(), model.skeleton))
                except Exception as e:  # noqa: BLE001
                    log.warning("postprocess failed for %s: %s", s.id, e)
            frames = self._to_frames(out)
        with s.lock:
            s.motion_tensor = samples
            s.frames = frames
            s.frame_offset = 0
            s.playhead = seeded
            s.gen_count += 1
        self.stats["windows"] += 1
        return seeded

    def _to_frames(self, out: dict) -> List[Frame]:
        lr = out["local_rot_mats"][0].detach().float().cpu().numpy()  # [G,J,3,3]
        gr = out["global_rot_mats"][0].detach().float().cpu().numpy()
        rp = out["root_positions"][0].detach().float().cpu().numpy()  # [G,3]
        fc = out["foot_contacts"][0].detach().cpu().numpy().astype(bool)  # [G,4]
        lq = mats_to_quats(lr)
        gq = mats_to_quats(gr)
        return [Frame(rp[i].astype(np.float32), lq[i], gq[i], fc[i]) for i in range(rp.shape[0])]

    # ---- background loop ------------------------------------------------------------
    def run_forever(self) -> None:
        while True:
            try:
                did = self._tick()
            except Exception:  # noqa: BLE001
                log.exception("generation tick failed")
                did = False
                time.sleep(0.1)
            if not did:
                self.wake.wait(0.02)
                self.wake.clear()

    def _tick(self) -> bool:
        with self.sessions_lock:
            sessions = list(self.sessions.values())
        did = False
        now = time.time()
        for s in sessions:
            if s.closed:
                continue
            if s.attached == 0 and now - s.last_seen > SESSION_IDLE_TIMEOUT:
                self.close_session(s.id)
                continue
            model = self.models.get(s.model_name)
            fps = float(model.motion_rep.fps) if model else 20.0
            lead = int(LEAD_SECONDS * fps)
            pending = s.pending_prompt
            if pending is not None and pending.strip() != s.prompt.strip():
                s.prompt = pending
                s.pending_prompt = None
                self.generate_step(s, replan=True)
                did = True
            elif pending is not None:
                s.pending_prompt = None
            elif s.burst and s.burst_frames > 0:
                if len(s.frames) < s.burst_frames + int(model.gen_horizon_len):
                    self.generate_step(s, replan=False)
                    did = True
            elif s.attached == 0:
                # Không ai đọc thì KHÔNG sinh. Phiên vẫn sống để client nối lại được (`last_seq`),
                # nhưng mỗi lần tải lại trang mà vẫn sinh tiếp thì các phiên bỏ rơi chia nhau GPU:
                # đo được 77 lần đói khung trên phiên đang xem vì hai phiên cũ vẫn chạy nền.
                continue
            elif not s.paused and (s.available() - s.playhead) < lead:
                self.generate_step(s, replan=False)
                did = True
        return did


ENGINE: Optional[Engine] = None


# --------------------------------------------------------------------------------------
# Frame streaming (shared by WebSocket and SSE)
# --------------------------------------------------------------------------------------
def build_frame_msg(s: Session, joints: List[str], seq: int, f0: Frame, f1: Optional[Frame], a: float) -> dict:
    if f1 is None or a <= 1e-6:
        pos, lq, gq, contacts = f0.root_pos, f0.local_q, f0.global_q, f0.contacts
    else:
        pos = (1.0 - a) * f0.root_pos + a * f1.root_pos
        lq = slerp(f0.local_q, f1.local_q, a) if s.space in ("local", "both") else f0.local_q
        gq = slerp(f0.global_q, f1.global_q, a) if s.space in ("global", "both") else f0.global_q
        contacts = f0.contacts if a < 0.5 else f1.contacts
    primary = gq if s.space == "global" else lq
    data: Dict[str, Any] = {}
    if s.fmt == "array":
        data["root"] = rnd(pos)
        data["quats"] = [rnd(q) for q in primary]
        if s.space == "both":
            data["quats_global"] = [rnd(q) for q in gq]
    else:
        bones = {}
        for j, name in enumerate(joints):
            q = rnd(primary[j])
            bones[name] = (rnd(pos) + q) if j == 0 else q
        data["bones"] = bones
        if s.space == "both":
            data["bones_global"] = {name: ((rnd(pos) + rnd(gq[j])) if j == 0 else rnd(gq[j])) for j, name in enumerate(joints)}
    return {
        "type": "motion_frame",
        "seq": seq,
        "t": round(seq / s.stream_fps, 4),
        "timestamp": round(time.time(), 3),
        "data": data,
        "contacts": [int(c) for c in contacts],
    }


async def stream_session(s: Session, send: Callable[[dict], Awaitable[None]], stop: asyncio.Event) -> None:
    """Emit frames at s.stream_fps with wall-clock pacing, interpolating between model frames."""
    engine = ENGINE
    model = engine.get_model(s.model_name)
    mfps = float(model.motion_rep.fps)
    joints = list(model.skeleton.bone_order_names)
    s.attached += 1
    s.last_seen = time.time()
    k0 = s.sent_count
    t_start = time.monotonic()
    buffering = False
    try:
        while not stop.is_set() and not s.closed:
            if s.paused:
                await asyncio.sleep(0.05)
                t_start = time.monotonic() - (s.sent_count - k0) / s.stream_fps
                continue
            k = s.sent_count
            if s.burst and s.burst_frames > 0 and (k * mfps / s.stream_fps) >= s.burst_frames:
                await send({"type": "status", "state": "done", "seq": k, "frames": k})
                break
            target = t_start + (k - k0) / s.stream_fps
            now = time.monotonic()
            if target > now and not s.burst:
                await asyncio.sleep(min(target - now, 0.25))
                continue
            pos_model = k * mfps / s.stream_fps
            i = int(math.floor(pos_model))
            a = pos_model - i
            need = i + 1 if a > 1e-6 else i
            with s.lock:
                f0 = s.get_frame(i)
                f1 = s.get_frame(i + 1) if a > 1e-6 else None
                ok = f0 is not None and (need == i or f1 is not None)
            if not ok:
                if not buffering:
                    buffering = True
                    s.underruns += 1
                    await send({"type": "status", "state": "buffering", "seq": k, "available": s.available()})
                engine.wake.set()
                await asyncio.sleep(0.02)
                # rebase the clock so we do not burst after the stall
                t_start = time.monotonic() - (k - k0) / s.stream_fps
                continue
            if buffering:
                buffering = False
                await send({"type": "status", "state": "streaming", "seq": k})
            s.playhead = i
            s.last_seen = time.time()
            if (s.available() - s.playhead) < int(LEAD_SECONDS * mfps):
                engine.wake.set()
            msg = build_frame_msg(s, joints, k, f0, f1, a)
            await send(msg)
            s.sent_count = k + 1
    finally:
        s.attached -= 1
        s.last_seen = time.time()


def rewind_to_seq(s: Session, last_seq: int) -> None:
    """Restart the stream right after `last_seq` if that frame is still in the session buffer."""
    model = ENGINE.get_model(s.model_name)
    mfps = float(model.motion_rep.fps)
    k = max(0, last_seq + 1)
    if k >= s.sent_count:
        return
    i = int(math.floor(k * mfps / s.stream_fps))
    with s.lock:
        if s.get_frame(i) is None:
            return
        skipped = s.sent_count - k
        s.sent_count = k
        s.playhead = i
    log.info("Session %s resume: rewound %d stream frames to seq %d (model frame %d)", s.id, skipped, k, i)


def session_info_msg(s: Session) -> dict:
    info = ENGINE.skeleton_info(s.model_name)
    info.update(
        {
            "type": "session",
            "session_id": s.id,
            "stream_fps": s.stream_fps,
            "space": s.space,
            "format": s.fmt,
            "history_frames": s.history_frames,
            "steps": s.steps,
            "cfg": list(s.cfg),
            "prompt": s.prompt,
            "resumed_at_seq": s.sent_count,
            "seeded_frames": s.seeded,
        }
    )
    return info


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
def create_app():
    app = FastAPI(title="ARDY motion stream")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/ping")
    async def ping():
        if ENGINE.ready:
            return {"status": "healthy"}
        if ENGINE.error:
            return JSONResponse({"status": "error", "error": ENGINE.error}, status_code=500)
        return Response(status_code=204)

    @app.get("/")
    @app.get("/health")
    async def health():
        with ENGINE.sessions_lock:
            sess = [
                {
                    "id": s.id,
                    "model": s.model_name,
                    "prompt": s.prompt,
                    "attached": s.attached,
                    "playhead": s.playhead,
                    "available": s.available(),
                    "windows": s.gen_count,
                    "replans": s.replan_count,
                    "last_gen_s": round(s.last_gen_time, 3),
                    "underruns": s.underruns,
                }
                for s in ENGINE.sessions.values()
            ]
        return {
            "ready": ENGINE.ready,
            "error": ENGINE.error,
            "device": ENGINE.device,
            "models": list(ENGINE.models),
            "text_source": (
                ("catalog:%d" % len(ENGINE.embeddings)) if ENGINE.embeddings is not None
                else ("none" if ENGINE.text_encoder is False else "encoder")
            ),
            "catalog_misses": dict(ENGINE.misses),
            "load_seconds": round(ENGINE.load_seconds, 1),
            "stats": ENGINE.stats,
            "sessions": sess,
        }

    @app.get("/skeleton")
    async def skeleton(model: str = DEFAULT_MODEL):
        if not ENGINE.ready:
            raise HTTPException(503, "loading")
        return ENGINE.skeleton_info(model)

    # ---- WebSocket --------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session: Optional[Session] = None
        sender: Optional[asyncio.Task] = None
        stop = asyncio.Event()
        send_lock = asyncio.Lock()

        async def send(msg: dict) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(msg, separators=(",", ":")))

        async def stop_sender() -> None:
            nonlocal sender
            stop.set()
            if sender is not None:
                try:
                    await asyncio.wait_for(sender, timeout=1.0)
                except Exception:  # noqa: BLE001
                    sender.cancel()
                sender = None

        try:
            if not ENGINE.ready:
                await send({"type": "error", "message": "engine not ready", "error": ENGINE.error})
                await ws.close(code=1013)
                return
            while True:
                # Clients must send something (a {"type":"ping"} is enough) at least every
                # CLIENT_TIMEOUT seconds. Gateways can keep a dead client's upstream socket open,
                # which would otherwise leave a session generating forever and hold a request slot.
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=CLIENT_TIMEOUT)
                except asyncio.TimeoutError:
                    log.info("ws idle for %ss, closing (session %s)", CLIENT_TIMEOUT, session.id if session else None)
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await send({"type": "error", "message": "invalid json"})
                    continue
                t = msg.get("type")
                if t == "start":
                    await stop_sender()
                    stop = asyncio.Event()
                    sid = msg.get("session_id")
                    session = ENGINE.get_session(sid) if sid else None
                    if session is not None and msg.get("last_seq") is not None:
                        # Resume from the last frame the client actually displayed: frames that were
                        # in flight through a gateway when the socket closed would otherwise be skipped.
                        rewind_to_seq(session, int(msg["last_seq"]))
                    if session is None:
                        try:
                            session = ENGINE.create_session(msg)
                        except RuntimeError as e:
                            await send({"type": "error", "message": str(e)})
                            continue
                    await send(session_info_msg(session))
                    sender = asyncio.create_task(stream_session(session, send, stop))
                elif t == "prompt":
                    if session is None:
                        await send({"type": "error", "message": "no session; send start first"})
                        continue
                    if "embedding" in msg:
                        try:
                            session.embedding = decode_embedding(msg.get("embedding"))
                        except Exception as e:  # noqa: BLE001
                            await send({"type": "error", "message": f"bad embedding: {e}"})
                            continue
                    session.pending_prompt = str(msg.get("text", ""))
                    ENGINE.wake.set()
                    await send({"type": "status", "state": "prompt_queued", "text": session.pending_prompt})
                elif t == "pause" and session:
                    session.paused = True
                elif t == "resume" and session:
                    session.paused = False
                    ENGINE.wake.set()
                elif t == "reset" and session:
                    ENGINE.reset_session(session)
                    await send({"type": "status", "state": "reset"})
                elif t == "stop":
                    await stop_sender()
                    if session:
                        ENGINE.close_session(session.id)
                        session = None
                    await send({"type": "status", "state": "stopped"})
                elif t == "ping":
                    await send({"type": "pong", "timestamp": round(time.time(), 3)})
                else:
                    await send({"type": "error", "message": f"unknown type {t!r}"})
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001
            log.warning("ws error: %s", e)
        finally:
            await stop_sender()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass
            # keep the session alive for SESSION_IDLE_TIMEOUT so a reconnect can resume it

    @app.post("/generate")
    async def generate(req: Request):
        """One cue in, one clip out. Stateless: safe behind a queue endpoint with many workers."""
        if not ENGINE.ready:
            raise HTTPException(503, "loading")
        body = await req.json()
        jobs = body.get("cues") or [body]
        try:
            clips = await asyncio.to_thread(lambda: [ENGINE.generate_clip(j) for j in jobs])
        except Exception as e:  # noqa: BLE001
            log.exception("generate failed")
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        return {"clips": clips} if body.get("cues") else clips[0]

    # ---- HTTP / SSE fallback ----------------------------------------------------------
    @app.post("/sessions")
    async def create_session(req: Request):
        if not ENGINE.ready:
            raise HTTPException(503, "loading")
        cfg = await req.json() if (await req.body()) else {}
        try:
            s = ENGINE.create_session(cfg)
        except RuntimeError as e:
            raise HTTPException(429, str(e))
        return session_info_msg(s)

    @app.get("/sessions/{sid}")
    async def get_session(sid: str):
        s = ENGINE.get_session(sid)
        if s is None:
            raise HTTPException(404, "no such session")
        return session_info_msg(s)

    @app.post("/sessions/{sid}/prompt")
    async def set_prompt(sid: str, req: Request):
        s = ENGINE.get_session(sid)
        if s is None:
            raise HTTPException(404, "no such session")
        body = await req.json()
        s.pending_prompt = str(body.get("text", ""))
        ENGINE.wake.set()
        return {"ok": True, "text": s.pending_prompt}

    @app.post("/sessions/{sid}/control")
    async def control(sid: str, req: Request):
        s = ENGINE.get_session(sid)
        if s is None:
            raise HTTPException(404, "no such session")
        action = (await req.json()).get("action")
        if action == "pause":
            s.paused = True
        elif action == "resume":
            s.paused = False
            ENGINE.wake.set()
        elif action == "reset":
            ENGINE.reset_session(s)
        elif action == "stop":
            ENGINE.close_session(sid)
        else:
            raise HTTPException(400, "unknown action")
        return {"ok": True}

    @app.delete("/sessions/{sid}")
    async def delete_session(sid: str):
        ENGINE.close_session(sid)
        return {"ok": True}

    @app.get("/sessions/{sid}/events")
    async def events(sid: str, max_frames: int = 0):
        s = ENGINE.get_session(sid)
        if s is None:
            raise HTTPException(404, "no such session")

        async def gen():
            queue: asyncio.Queue = asyncio.Queue(maxsize=64)
            stop = asyncio.Event()

            async def send(msg: dict) -> None:
                await queue.put(msg)

            task = asyncio.create_task(stream_session(s, send, stop))
            n = 0
            try:
                yield "data: " + json.dumps(session_info_msg(s), separators=(",", ":")) + "\n\n"
                while not stop.is_set():
                    msg = await queue.get()
                    yield "data: " + json.dumps(msg, separators=(",", ":")) + "\n\n"
                    if msg.get("type") == "motion_frame":
                        n += 1
                        if max_frames and n >= max_frames:
                            break
            finally:
                stop.set()
                task.cancel()

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


# --------------------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------------------
def selftest(windows: int = 6) -> int:
    global ENGINE
    ENGINE = Engine()
    ENGINE.load()
    if not ENGINE.ready:
        print("SELFTEST FAILED:", ENGINE.error)
        return 1
    s = ENGINE.create_session({"prompt": "A person is walking.", "stream_fps": 30, "space": "both"})
    model = ENGINE.get_model(s.model_name)
    fps = float(model.motion_rep.fps)
    times = []
    for i in range(windows):
        replan = i == windows // 2
        if replan:
            s.prompt = "A person jumps."
            s.playhead = max(0, s.available() - 6)
        ENGINE.generate_step(s, replan=replan)
        times.append(s.last_gen_time)
    joints = list(model.skeleton.bone_order_names)
    f0, f1 = s.frames[10], s.frames[11]
    msg = build_frame_msg(s, joints, 15, f0, f1, 0.5)
    payload = json.dumps(msg, separators=(",", ":"))
    print("SELFTEST windows:", windows, "gen times:", [round(t, 3) for t in times])
    print(
        "SELFTEST realtime factor (window %.2fs):" % (model.gen_horizon_len / fps),
        round((model.gen_horizon_len / fps) / (sum(times[1:]) / max(len(times) - 1, 1)), 2),
    )
    print("SELFTEST frames available:", s.available(), "payload bytes/frame:", len(payload))
    print("SELFTEST sample hips:", msg["data"]["bones"]["Hips"])
    print("SELFTEST root y range:", round(min(f.root_pos[1] for f in s.frames), 3), round(max(f.root_pos[1] for f in s.frames), 3))
    print("SELFTEST skeleton:", json.dumps(ENGINE.skeleton_info(s.model_name))[:400], "...")
    ENGINE.close_session(s.id)
    return 0


def main() -> None:
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest(args.windows))

    import uvicorn

    ENGINE = Engine()
    threading.Thread(target=ENGINE.load, name="loader", daemon=True).start()
    threading.Thread(target=ENGINE.run_forever, name="generator", daemon=True).start()
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info", ws_ping_interval=20.0, ws_ping_timeout=20.0)


if __name__ == "__main__":
    main()
