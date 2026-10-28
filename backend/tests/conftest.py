import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(request):
    """Rate limits (and the cached training probe) live in the cache.

    Tests use an in-memory cache instead of the file cache in the data dir, so they neither
    touch a running dev server's cache nor disturb other test runs, and every test starts
    with a clean slate.
    """
    try:
        settings = request.getfixturevalue("settings")
        from django.core.cache import cache
    except Exception:  # pragma: no cover - Django / pytest-django not available
        yield
        return
    settings.CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "brickdnn-tests",
        }
    }
    cache.clear()
    yield
