"""Embedding catalog: sentence -> ARDY text vector, without running the 16 GB encoder online.

    from catalog import CatalogService
    svc = CatalogService("file://~/ardy-catalog")
    hit = svc.lookup("A person waves with the right hand.")
    # hit.vector -> float32[4096], send it to the ARDY worker with the generation job

Storage is an S3-shaped object store (`catalog.store`): a local directory during development,
Cloudflare R2 in production, same keys either way.
"""

from .service import CatalogService, Lookup, MiniLM
from .store import LocalStore, ObjectStore, R2Store, norm_sentence, sentence_hash

__all__ = [
    "CatalogService",
    "Lookup",
    "MiniLM",
    "ObjectStore",
    "LocalStore",
    "R2Store",
    "norm_sentence",
    "sentence_hash",
]
