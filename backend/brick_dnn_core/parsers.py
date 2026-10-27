from rest_framework.exceptions import ParseError
from rest_framework.parsers import JSONParser


class SafeJSONParser(JSONParser):
    """JSON parser that rejects absurdly nested documents with a 400 instead of crashing."""

    def parse(self, stream, media_type=None, parser_context=None):
        try:
            return super().parse(stream, media_type, parser_context)
        except RecursionError as exc:
            raise ParseError("JSON is nested too deeply.") from exc
