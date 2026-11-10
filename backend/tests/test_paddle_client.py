"""Paddle (paddle.py): webhook signatures, and the API client against a small HTTP server on
this machine."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from brick_dnn_core import paddle
from brick_dnn_core.paddle import (
    PaddleClient,
    PaddleError,
    PaddleUnavailable,
    SignatureError,
    parse_signature,
    sign,
    signature_header,
    verify_signature,
)

SECRET = "pdl_ntfset_test_secret"


# --------------------------------------------------------------------------- #
# Signatures (paddle.py)
# --------------------------------------------------------------------------- #

BODY = b'{"event_id":"evt_1","event_type":"subscription.updated"}'


def test_signature_is_hmac_sha256_of_timestamp_and_raw_body():
    expected = hmac.new(SECRET.encode(), b"1700000000:" + BODY, hashlib.sha256).hexdigest()
    assert sign(BODY, SECRET, 1700000000) == expected
    header = f"ts=1700000000;h1={expected}"
    assert signature_header(BODY, SECRET, 1700000000) == header
    assert parse_signature(header) == (1700000000, [expected])
    assert verify_signature(BODY, header, [SECRET], tolerance=5, now=1700000003) == 1700000000


def test_signature_of_a_changed_body_or_another_secret_is_refused():
    header = signature_header(BODY, SECRET)
    with pytest.raises(SignatureError, match="does not match"):
        verify_signature(BODY + b" ", header, [SECRET], tolerance=300)
    with pytest.raises(SignatureError, match="does not match"):
        verify_signature(BODY, header, ["another-secret"], tolerance=300)
    with pytest.raises(SignatureError, match="does not match"):
        verify_signature(BODY, header, [], tolerance=300)
    with pytest.raises(SignatureError, match="does not match"):
        verify_signature(BODY, header, [""], tolerance=300)


def test_old_and_future_signatures_are_refused():
    now = time.time()
    old = signature_header(BODY, SECRET, int(now) - 301)
    with pytest.raises(SignatureError, match="seconds old"):
        verify_signature(BODY, old, [SECRET], tolerance=300, now=now)
    future = signature_header(BODY, SECRET, int(now) + 301)
    with pytest.raises(SignatureError, match="seconds old"):
        verify_signature(BODY, future, [SECRET], tolerance=300, now=now)
    assert verify_signature(BODY, old, [SECRET], tolerance=0, now=now)  # 0: any age
    recent = signature_header(BODY, SECRET, int(now) - 200)
    assert verify_signature(BODY, recent, [SECRET], tolerance=300, now=now)


@pytest.mark.parametrize(
    "header",
    [None, "", "nonsense", "ts=1700000000", "h1=abc", "ts=soon;h1=abc", "ts=;h1="],
)
def test_missing_or_malformed_signature_headers_are_refused(header):
    with pytest.raises(SignatureError):
        verify_signature(BODY, header, [SECRET], tolerance=0)


def test_signatures_while_a_secret_is_rotated():
    timestamp = int(time.time())
    new = sign(BODY, "new-secret", timestamp)
    old = sign(BODY, SECRET, timestamp)
    # Two signatures in the header, or two secrets on the server.
    assert verify_signature(BODY, f"ts={timestamp};h1={new};h1={old}", [SECRET], 300)
    assert verify_signature(BODY, f"ts={timestamp};h1={new}", [SECRET, "new-secret"], 300)


# --------------------------------------------------------------------------- #
# The API client
# --------------------------------------------------------------------------- #


class Paddle(HTTPServer):
    """Answers every request with the next of ``answers`` (status, JSON) and keeps what was
    asked in ``requests``."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.answers: list[tuple[int, dict]] = []
        self.requests: list[dict] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    server: Paddle

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(body) if body else None,
            }
        )
        status, payload = self.server.answers.pop(0) if self.server.answers else (200, {"data": {}})
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = do_PATCH = _answer

    def log_message(self, *args) -> None:  # quiet
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(paddle.time, "sleep", lambda seconds: None)
    instance = Paddle()
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()


@pytest.fixture
def client(server):
    return PaddleClient("pdl_sdbx_apikey_test", "sandbox", base_url=server.url)


def error(code: str, detail: str = "No.") -> dict:
    return {"error": {"type": "request_error", "code": code, "detail": detail}}


def test_addresses_of_the_two_environments():
    assert PaddleClient("key", "sandbox").base_url == "https://sandbox-api.paddle.com"
    assert PaddleClient("key", "live").base_url == "https://api.paddle.com"


def test_requests_carry_the_key_and_the_api_version(server, client):
    server.answers = [(201, {"data": {"id": "ctm_1", "email": "ada@example.com"}})]
    assert client.create_customer("ada@example.com", {"user_id": "7"})["id"] == "ctm_1"
    (request,) = server.requests
    assert (request["method"], request["path"]) == ("POST", "/customers")
    assert request["headers"]["Authorization"] == "Bearer pdl_sdbx_apikey_test"
    assert request["headers"]["Paddle-Version"] == "1"
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["body"] == {"email": "ada@example.com", "custom_data": {"user_id": "7"}}


def test_the_calls_billing_makes(server, client):
    server.answers = [
        (200, {"data": [{"id": "ctm_1"}]}),
        (200, {"data": {"id": "sub_1"}}),
        (200, {"data": [{"id": "sub_1"}]}),
        (200, {"data": {"id": "sub_1"}}),
        (200, {"data": {"id": "sub_1", "update_summary": {}}}),
        (200, {"data": {"id": "sub_1"}}),
        (200, {"data": {"id": "sub_1"}}),
        (200, {"data": {"id": "sub_1"}}),
        (201, {"data": {"urls": {"general": {"overview": "https://portal.example/x"}}}}),
        (200, {"data": {"id": "ctm_1"}}),
        (200, {"data": {"id": "pri_1"}}),
        (200, {"data": {"id": "pro_1"}}),
    ]
    assert client.find_customers("a+b@example.com") == [{"id": "ctm_1"}]
    client.get_subscription("sub_1")
    client.list_subscriptions("ctm_1")
    client.change_price("sub_1", "pri_2", "prorated_immediately")
    client.change_price("sub_1", "pri_2", "prorated_next_billing_period", preview=True)
    client.cancel_subscription("sub_1", immediately=False)
    client.cancel_subscription("sub_1", immediately=True)
    client.remove_scheduled_change("sub_1")
    assert client.create_portal_session("ctm_1", ["sub_1"]) == "https://portal.example/x"
    client.update_customer("ctm_1", email="new@example.com")
    client.get_price("pri_1")
    client.get_product("pro_1")
    asked = [(r["method"], r["path"], r["body"]) for r in server.requests]
    items = [{"price_id": "pri_2", "quantity": 1}]
    assert asked == [
        ("GET", "/customers?email=a%2Bb%40example.com&status=active%2Carchived", None),
        ("GET", "/subscriptions/sub_1", None),
        ("GET", "/subscriptions?customer_id=ctm_1&per_page=50", None),
        (
            "PATCH",
            "/subscriptions/sub_1",
            {"items": items, "proration_billing_mode": "prorated_immediately"},
        ),
        (
            "PATCH",
            "/subscriptions/sub_1/preview",
            {"items": items, "proration_billing_mode": "prorated_next_billing_period"},
        ),
        ("POST", "/subscriptions/sub_1/cancel", {"effective_from": "next_billing_period"}),
        ("POST", "/subscriptions/sub_1/cancel", {"effective_from": "immediately"}),
        ("PATCH", "/subscriptions/sub_1", {"scheduled_change": None}),
        ("POST", "/customers/ctm_1/portal-sessions", {"subscription_ids": ["sub_1"]}),
        ("PATCH", "/customers/ctm_1", {"email": "new@example.com"}),
        ("GET", "/prices/pri_1", None),
        ("GET", "/products/pro_1", None),
    ]


def test_one_time_charge_is_a_price_made_for_it_with_tax_included(server, client):
    client.create_charge(
        "sub_1",
        product_id="pro_overage",
        description="Brick DNN: 3.25 CPU hours beyond the Plus plan in October 2026",
        amount_minor=130,
        currency="EUR",
        immediately=True,
        custom_data={"brickdnn_overage_charge": "5"},
    )
    (request,) = server.requests
    assert (request["method"], request["path"]) == ("POST", "/subscriptions/sub_1/charge")
    assert request["body"] == {
        "effective_from": "immediately",
        "items": [
            {
                "quantity": 1,
                "price": {
                    "product_id": "pro_overage",
                    "description": "Brick DNN: 3.25 CPU hours beyond the Plus plan in October 2026",
                    "name": "Brick DNN: 3.25 CPU hours beyond the Plus plan in October 2026",
                    "unit_price": {"amount": "130", "currency_code": "EUR"},
                    "tax_mode": "internal",
                    "custom_data": {"brickdnn_overage_charge": "5"},
                },
            }
        ],
    }
    client.create_charge(
        "sub_1",
        product_id="pro_overage",
        description="x",
        amount_minor=50,
        currency="EUR",
        immediately=False,
    )
    assert server.requests[1]["body"]["effective_from"] == "next_billing_period"


def test_a_refusal_carries_paddles_error_code(server, client):
    server.answers = [(409, error("customer_already_exists", "customer email conflicts"))]
    with pytest.raises(PaddleError) as caught:
        client.create_customer("ada@example.com", {})
    assert (caught.value.status, caught.value.code) == (409, "customer_already_exists")
    assert not isinstance(caught.value, PaddleUnavailable)
    assert "customer email conflicts" in str(caught.value)
    assert "pdl_sdbx_apikey_test" not in str(caught.value)


def test_changes_are_never_repeated(server, client):
    server.answers = [(500, error("internal_error")), (200, {"data": {"id": "sub_1"}})]
    with pytest.raises(PaddleUnavailable):
        client.create_charge(
            "sub_1",
            product_id="pro_1",
            description="x",
            amount_minor=100,
            currency="EUR",
            immediately=True,
        )
    assert len(server.requests) == 1  # asked once: whether it was billed is unknown


def test_reading_is_tried_once_more_when_paddle_is_busy(server, client):
    server.answers = [(429, error("too_many_requests")), (200, {"data": {"id": "sub_1"}})]
    assert client.get_subscription("sub_1") == {"id": "sub_1"}
    assert len(server.requests) == 2
    server.answers = [(503, {}), (502, {})]
    with pytest.raises(PaddleUnavailable):
        client.get_subscription("sub_1")
    assert len(server.requests) == 4


def test_no_answer_and_unreadable_answers_are_unavailable(server, client):
    server.answers = [(200, {"unexpected": True})]
    with pytest.raises(PaddleUnavailable, match="could not be read"):
        client.cancel_subscription("sub_1", immediately=True)
    nobody = PaddleClient("key", "sandbox", base_url="http://127.0.0.1:9")
    with pytest.raises(PaddleUnavailable, match="did not answer"):
        nobody.cancel_subscription("sub_1", immediately=True)
