import pytest


@pytest.fixture(autouse=True)
def _clear_throttle_cache():
    """Rate limits are stored in the cache; start every test with a clean slate."""
    try:
        from django.core.cache import cache
    except Exception:  # pragma: no cover - Django not configured
        yield
        return
    cache.clear()
    yield
