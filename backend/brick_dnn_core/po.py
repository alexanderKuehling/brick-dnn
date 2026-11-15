"""Reading and writing gettext catalogues (``.po``) and compiling them (``.mo``) in Python.

Django's ``makemessages`` and ``compilemessages`` need GNU gettext, which neither the Docker
image nor CI has. The format is simple, so ``manage.py update_translations`` (see
docs/translations.md) uses this module instead: it collects the strings, keeps the ``.po``
file up to date and writes the ``.mo`` file Django reads, which is committed.
"""

from __future__ import annotations

import ast
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entry:
    msgid: str
    msgstr: str = ""
    msgid_plural: str = ""
    msgstr_plural: dict[int, str] = field(default_factory=dict)
    msgctxt: str | None = None
    comments: list[str] = field(default_factory=list)
    fuzzy: bool = False

    @property
    def key(self) -> tuple[str | None, str]:
        return (self.msgctxt, self.msgid)

    @property
    def translated(self) -> bool:
        if self.fuzzy:
            return False
        if self.msgid_plural:
            return bool(self.msgstr_plural) and all(self.msgstr_plural.values())
        return bool(self.msgstr)


@dataclass
class Catalog:
    header: str = ""
    entries: list[Entry] = field(default_factory=list)

    def get(self, msgid: str, msgctxt: str | None = None) -> Entry | None:
        for entry in self.entries:
            if entry.key == (msgctxt, msgid):
                return entry
        return None


def _unquote(text: str) -> str:
    return ast.literal_eval(text) if text else ""


def parse(text: str) -> Catalog:
    """A ``.po`` file as a Catalog; the entry with an empty msgid becomes the header."""
    catalog = Catalog()
    entries: list[Entry] = []
    entry: Entry | None = None
    current = ""
    index = 0
    comments: list[str] = []
    fuzzy = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("#,") and "fuzzy" in line:
                fuzzy = True
            elif line.startswith(("#.", "#:")):
                comments.append(line)
            continue
        match = re.match(r'^(msgctxt|msgid_plural|msgid|msgstr(?:\[(\d+)\])?)\s+(".*")$', line)
        if match:
            keyword, number, value = match.group(1), match.group(2), _unquote(match.group(3))
            starts_entry = keyword == "msgctxt" or (keyword == "msgid" and current != "msgctxt")
            if starts_entry:
                entry = Entry(msgid="", comments=comments, fuzzy=fuzzy)
                entries.append(entry)
                comments, fuzzy = [], False
            if entry is None:
                continue
            if keyword == "msgctxt":
                entry.msgctxt = value
            elif keyword == "msgid":
                entry.msgid = value
            elif keyword == "msgid_plural":
                entry.msgid_plural = value
            elif number is not None:
                index = int(number)
                entry.msgstr_plural[index] = value
                keyword = "msgstr_plural"
            else:
                entry.msgstr = value
            current = keyword
            continue
        if line.startswith('"') and entry is not None:
            value = _unquote(line)
            if current == "msgctxt":
                entry.msgctxt = (entry.msgctxt or "") + value
            elif current == "msgid":
                entry.msgid += value
            elif current == "msgid_plural":
                entry.msgid_plural += value
            elif current == "msgstr":
                entry.msgstr += value
            elif current == "msgstr_plural":
                entry.msgstr_plural[index] += value
    for found in entries:
        if found.msgid == "" and found.msgctxt is None:
            catalog.header = found.msgstr
        else:
            catalog.entries.append(found)
    return catalog


def _quote(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _field(keyword: str, text: str) -> list[str]:
    if "\n" not in text.rstrip("\n") and len(text) < 70:
        return [f"{keyword} {_quote(text)}"]
    parts = re.findall(r"[^\n]*\n|[^\n]+$", text) or [""]
    return [f'{keyword} ""', *(_quote(part) for part in parts)]


def dump(catalog: Catalog) -> str:
    header_lines = re.findall(r"[^\n]*\n|[^\n]+$", catalog.header)
    lines = ['msgid ""', 'msgstr ""', *(_quote(part) for part in header_lines)]
    for entry in catalog.entries:
        lines.append("")
        lines.extend(entry.comments)
        if entry.fuzzy:
            lines.append("#, fuzzy")
        if entry.msgctxt is not None:
            lines.extend(_field("msgctxt", entry.msgctxt))
        lines.extend(_field("msgid", entry.msgid))
        if entry.msgid_plural:
            lines.extend(_field("msgid_plural", entry.msgid_plural))
            for number in sorted(entry.msgstr_plural) or [0, 1]:
                lines.extend(_field(f"msgstr[{number}]", entry.msgstr_plural.get(number, "")))
        else:
            lines.extend(_field("msgstr", entry.msgstr))
    return "\n".join(lines) + "\n"


def compile_mo(catalog: Catalog) -> bytes:
    """The ``.mo`` file of ``catalog``: translated entries only, as GNU msgfmt writes them."""
    messages: dict[bytes, bytes] = {b"": catalog.header.encode("utf-8")}
    for entry in catalog.entries:
        if not entry.translated:
            continue
        msgid = entry.msgid
        if entry.msgctxt is not None:
            msgid = f"{entry.msgctxt}\x04{msgid}"
        if entry.msgid_plural:
            key = f"{msgid}\x00{entry.msgid_plural}".encode()
            value = "\x00".join(
                entry.msgstr_plural[n] for n in sorted(entry.msgstr_plural)
            ).encode()
        else:
            key, value = msgid.encode(), entry.msgstr.encode()
        messages[key] = value
    keys = sorted(messages)
    ids = b""
    strs = b""
    offsets = []
    for key in keys:
        offsets.append((len(ids), len(key), len(strs), len(messages[key])))
        ids += key + b"\x00"
        strs += messages[key] + b"\x00"
    count = len(keys)
    key_start = 7 * 4
    value_start = key_start + count * 8
    ids_start = value_start + count * 8
    strs_start = ids_start + len(ids)
    key_table = []
    value_table = []
    for id_offset, id_length, str_offset, str_length in offsets:
        key_table += [id_length, ids_start + id_offset]
        value_table += [str_length, strs_start + str_offset]
    header = struct.pack("Iiiiiii", 0x950412DE, 0, count, key_start, value_start, 0, 0)
    tables = struct.pack(f"{len(key_table)}i", *key_table) + struct.pack(
        f"{len(value_table)}i", *value_table
    )
    return header + tables + ids + strs


def read(path: Path) -> Catalog:
    return parse(path.read_text(encoding="utf-8")) if path.exists() else Catalog()
