"""Error monitoring with Sentry: off without a DSN, and nothing personal when on."""

from __future__ import annotations

from brick_dnn_site import monitoring

DSN = "https://key@o1.ingest.de.sentry.io/2"


def test_off_without_a_dsn(monkeypatch):
    assert monitoring.options({}) is None
    assert monitoring.options({"BRICKDNN_SENTRY_DSN": "  "}) is None
    started = []
    monkeypatch.setattr("sentry_sdk.init", lambda **kwargs: started.append(kwargs))
    assert monitoring.init_sentry({}) is False
    assert started == []


def test_options_send_nothing_personal():
    found = monitoring.options({"BRICKDNN_SENTRY_DSN": DSN})
    assert found["dsn"] == DSN
    assert found["send_default_pii"] is False
    assert found["max_request_body_size"] == "never"
    assert found["environment"] == "production"
    assert found["release"] is None


def test_environment_release_and_rate_from_the_settings():
    found = monitoring.options(
        {
            "BRICKDNN_SENTRY_DSN": DSN,
            "BRICKDNN_SENTRY_ENVIRONMENT": "staging",
            "BRICKDNN_SENTRY_RELEASE": "brick-dnn@abc123",
            "BRICKDNN_SENTRY_TRACES_SAMPLE_RATE": "0.25",
        }
    )
    assert (found["environment"], found["release"]) == ("staging", "brick-dnn@abc123")
    assert found["traces_sampler"]({"wsgi_environ": {"PATH_INFO": "/api/projects/"}}) == 0.25


def test_traces_sampler_defaults_and_health_checks():
    sampler = monitoring.options({"BRICKDNN_SENTRY_DSN": DSN})["traces_sampler"]
    assert sampler({"wsgi_environ": {"PATH_INFO": "/api/projects/"}}) == 0.1
    assert sampler({"asgi_scope": {"path": "/api/health/"}}) == 0.0
    # A trace the browser started is followed either way.
    assert sampler({"parent_sampled": True}) == 1.0
    assert sampler({"parent_sampled": False}) == 0.0
    debug = monitoring.options({"BRICKDNN_SENTRY_DSN": DSN}, debug=True)
    assert debug["environment"] == "development"
    assert debug["traces_sampler"]({}) == 1.0


def test_bad_rates_fall_back_and_are_clamped():
    def rate(value):
        found = monitoring.options(
            {"BRICKDNN_SENTRY_DSN": DSN, "BRICKDNN_SENTRY_TRACES_SAMPLE_RATE": value}
        )
        return found["traces_sampler"]({})

    assert rate("lots") == 0.1
    assert rate("7") == 1.0
    assert rate("-1") == 0.0


def test_init_passes_the_options_and_ignores_access_logs(monkeypatch):
    started, ignored = [], []
    monkeypatch.setattr("sentry_sdk.init", lambda **kwargs: started.append(kwargs))
    monkeypatch.setattr("sentry_sdk.integrations.logging.ignore_logger", ignored.append)
    assert monitoring.init_sentry({"BRICKDNN_SENTRY_DSN": DSN}) is True
    assert started[0]["dsn"] == DSN and started[0]["send_default_pii"] is False
    assert "django.channels.server" in ignored
