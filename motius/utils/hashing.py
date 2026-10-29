"""Streaming content digests for model and data provenance."""
import hashlib
from pathlib import Path

def sha256_file(path,*,chunk_bytes=8*1024*1024):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(chunk_bytes),b''):digest.update(chunk)
    return digest.hexdigest()
