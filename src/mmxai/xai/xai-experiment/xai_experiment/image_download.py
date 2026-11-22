"""Download review images with checked content and recorded provenance.

COCO publishes its images in gs://images.cocodataset.org. The path-style
storage.googleapis.com URL avoids certificates tied to the bucket hostname.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


_PUBLIC_IMAGE_HOSTS = {"images.cocodataset.org", "storage.googleapis.com", "cs.stanford.edu"}


def _candidates(source_url: str) -> tuple[str, ...]:
    parts = urlsplit(source_url)
    if parts.scheme not in {"http", "https"} or parts.hostname not in _PUBLIC_IMAGE_HOSTS:
        raise ValueError(f"unsupported review-image URL: {source_url!r}")
    if parts.hostname == "images.cocodataset.org":
        bucket = urlunsplit(("https", "storage.googleapis.com",
                              "/images.cocodataset.org" + parts.path, parts.query, ""))
        return (bucket, source_url)
    return (source_url,)


def _is_jpeg(data: bytes) -> bool:
    return len(data) > 100 and data.startswith(b"\xff\xd8") and data.rstrip().endswith(b"\xff\xd9")


def fetch_review_image(source_url: str, destination: Path, *,
                       allow_unverified_tls: bool = False, session=None) -> dict:
    """Save a JPEG atomically and return source, actual URL, TLS state and hash.

    Verified HTTPS candidates are always tried first. An optional last resort
    accepts an unverified certificate only for the known public image hosts;
    the return record clearly marks that result for human review.
    """
    import certifi
    import requests

    destination = Path(destination)
    candidates = _candidates(source_url)
    if destination.is_file() and _is_jpeg(destination.read_bytes()):
        data = destination.read_bytes()
        return {"source_url": source_url, "download_url": "cached_existing_file",
                "tls_verified": "unknown_cached", "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data), "status": "cached"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    requester = session or requests.Session()
    failures = []
    attempts = [(url, certifi.where()) for url in candidates]
    if allow_unverified_tls:
        attempts += [(url, False) for url in candidates]
    for url, verification in attempts:
        try:
            if verification is False:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning)
                    response = requester.get(url, timeout=(10, 30), verify=False,
                                             headers={"User-Agent": "graphml-xai-review/1.0"})
            else:
                response = requester.get(url, timeout=(10, 30), verify=verification,
                                         headers={"User-Agent": "graphml-xai-review/1.0"})
            response.raise_for_status()
            final_url = response.url
            if urlsplit(final_url).hostname not in _PUBLIC_IMAGE_HOSTS:
                raise ValueError(f"redirected outside known image hosts: {final_url}")
            data = response.content
            if not _is_jpeg(data):
                raise ValueError(f"response is not a complete JPEG ({len(data)} bytes)")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            return {"source_url": source_url, "download_url": final_url,
                    "tls_verified": verification is not False,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data), "status": "downloaded"}
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"{url} ({'verified' if verification else 'unverified'}): {exc}")
    raise RuntimeError("; ".join(failures))
