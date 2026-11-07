"""The optional .env file in the repository root."""

import os

from brick_dnn_site.env_file import load_env_file, parse_env


def test_parse_env():
    text = """
# Database
DATABASE_URL=postgresql://postgres.ref:s3cr%23t@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
export DJANGO_DEBUG=true   # comment
QUOTED="a # b"
QUOTED_WITH_COMMENT="x"  # comment
SINGLE='$not_expanded'
EMPTY=
ONLY_COMMENT=            # empty: generated
HASH=#literal
  SPACED = value with spaces
not a line
1BAD=x
"""
    assert parse_env(text) == {
        "DATABASE_URL": (
            "postgresql://postgres.ref:s3cr%23t@aws-0-eu-central-1.pooler.supabase.com:5432/postgres"
        ),
        "DJANGO_DEBUG": "true",
        "QUOTED": "a # b",
        "QUOTED_WITH_COMMENT": "x",
        "SINGLE": "$not_expanded",
        "EMPTY": "",
        "ONLY_COMMENT": "",
        "HASH": "#literal",
        "SPACED": "value with spaces",
    }


def test_load_env_file_keeps_variables_that_are_set(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("BRICKDNN_TEST_A=from-file\nBRICKDNN_TEST_B=from-file\n")
    monkeypatch.setenv("BRICKDNN_TEST_A", "from-shell")
    monkeypatch.setenv("BRICKDNN_TEST_B", "")  # so that monkeypatch removes it afterwards
    monkeypatch.delenv("BRICKDNN_TEST_B")
    load_env_file(env)
    assert os.environ["BRICKDNN_TEST_A"] == "from-shell"
    assert os.environ["BRICKDNN_TEST_B"] == "from-file"


def test_no_env_file(tmp_path):
    load_env_file(tmp_path / ".env")  # nothing happens
