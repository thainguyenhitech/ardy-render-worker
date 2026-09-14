"""Object store for the embedding catalog: local filesystem now, Cloudflare R2 later.

The key layout is deliberately R2/S3-shaped so switching backends is a one-line change:

    catalog/manifest.json          {"count", "encoder", "dim", "updated"}
    catalog/index.npz              sentences + hashes + MiniLM vectors (backend loads all of it)
    catalog/vec/<hash>.f16         one 8 KB LLM2Vec vector per sentence (fetched on demand)
    catalog/misses.jsonl           sentences seen but not yet encoded

Local:  ObjectStore.open("file:///path/to/catalog")
R2:     ObjectStore.open("r2://bucket/prefix")   (needs R2_ACCOUNT_ID / R2_ACCESS_KEY_ID /
                                                  R2_SECRET_ACCESS_KEY, and boto3 installed)
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from typing import Iterator, Optional

__all__ = ["ObjectStore", "LocalStore", "R2Store", "norm_sentence", "sentence_hash"]


def norm_sentence(s: str) -> str:
    """Catalog key: case- and whitespace-insensitive, punctuation kept."""
    return re.sub(r"\s+", " ", s.strip()).lower()


def sentence_hash(s: str) -> str:
    """Stable 16-hex-char id for a sentence (hash of its normalized form)."""
    return hashlib.sha256(norm_sentence(s).encode("utf-8")).hexdigest()[:16]


class ObjectStore:
    """Minimal get/put/list/delete over an S3-like namespace."""

    @staticmethod
    def open(url: str) -> "ObjectStore":
        if url.startswith("r2://") or url.startswith("s3://"):
            return R2Store(url)
        path = url[len("file://") :] if url.startswith("file://") else url
        return LocalStore(path)

    # -- interface ---------------------------------------------------------------------
    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def append(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def list(self, prefix: str = "") -> Iterator[str]:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------------------
    def get_text(self, key: str, default: str = "") -> str:
        raw = self.get(key)
        return default if raw is None else raw.decode("utf-8")

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class LocalStore(ObjectStore):
    """Filesystem stand-in for R2. Same keys, same semantics, no network."""

    def __init__(self, root: str):
        self.root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"unsafe key: {key!r}")
        return os.path.join(self.root, key)

    def get(self, key: str) -> Optional[bytes]:
        try:
            with open(self._path(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # atomic replace, so a reader never sees a half-written object (R2 puts are atomic too)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def append(self, key: str, data: bytes) -> None:
        """R2 has no append; the real store emulates it with one object per line."""
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "ab") as f:
            f.write(data)

    def list(self, prefix: str = "") -> Iterator[str]:
        base = os.path.join(self.root, prefix)
        start = os.path.dirname(base) if not os.path.isdir(base) else base
        for dirpath, _dirs, files in os.walk(start):
            for name in files:
                key = os.path.relpath(os.path.join(dirpath, name), self.root)
                if key.startswith(prefix):
                    yield key

    def delete(self, key: str) -> None:
        try:
            os.unlink(self._path(key))
        except FileNotFoundError:
            pass

    def nuke(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.root, exist_ok=True)

    def __repr__(self) -> str:
        return f"LocalStore({self.root})"


class R2Store(ObjectStore):
    """Cloudflare R2 (S3-compatible). Only used in production; import stays lazy."""

    def __init__(self, url: str):
        rest = url.split("://", 1)[1]
        self.bucket, _, self.prefix = rest.partition("/")
        self.prefix = self.prefix.rstrip("/")
        import boto3  # noqa: PLC0415  (optional dependency)

        account = os.environ["R2_ACCOUNT_ID"]
        self.s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str) -> Optional[bytes]:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def append(self, key: str, data: bytes) -> None:
        cur = self.get(key) or b""
        self.put(key, cur + data)

    def list(self, prefix: str = "") -> Iterator[str]:
        token, full = None, self._key(prefix)
        while True:
            kw = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kw["ContinuationToken"] = token
            r = self.s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                k = o["Key"]
                yield k[len(self.prefix) + 1 :] if self.prefix else k
            if not r.get("IsTruncated"):
                return
            token = r.get("NextContinuationToken")

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self._key(key))

    def __repr__(self) -> str:
        return f"R2Store({self.bucket}/{self.prefix})"
