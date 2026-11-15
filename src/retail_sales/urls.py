"""URL helpers for configuring the frontend from environment variables."""


def normalize_base_url(raw: str) -> str:
    """Tolerate copy-paste mistakes in a base URL: quotes, stray characters, no scheme, trailing slash."""
    # A URL has no spaces, so other tokens (e.g. a copied table border "│") are junk.
    tokens = raw.replace('"', " ").replace("'", " ").split()
    url = next((t for t in tokens if any(c.isalnum() for c in t)), "").rstrip("/")
    if url and "://" not in url:
        url = f"https://{url}"
    return url
