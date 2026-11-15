"""Collect the strings to translate into ``locale/<language>/LC_MESSAGES/django.po`` and compile
the ``.mo`` file Django reads (docs/translations.md). Needs no GNU gettext.

    python manage.py update_translations           # update the .po and .mo files
    python manage.py update_translations --check   # fail if a string has no translation
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core import translations


class Command(BaseCommand):
    help = "Update the gettext catalogues (.po) and compile them (.mo)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--check",
            action="store_true",
            help="Change nothing; fail if the .po file is out of date or a string is untranslated.",
        )

    def handle(self, *args, check: bool = False, **options):
        messages = translations.collect()
        problems = []
        for language in translations.LANGUAGES:
            catalog, missing = translations.update(language, messages)
            if check:
                path = translations.po_path(language)
                current = path.read_text(encoding="utf-8") if path.exists() else ""
                from brick_dnn_core import po

                if current != po.dump(catalog):
                    problems.append(f"{path} is out of date: run manage.py update_translations.")
                mo = path.with_suffix(".mo")
                if not mo.exists() or mo.read_bytes() != po.compile_mo(catalog):
                    problems.append(f"{mo} is out of date: run manage.py update_translations.")
            else:
                translations.write(language, catalog)
                self.stdout.write(
                    f"{language}: {len(catalog.entries)} strings, {len(missing)} untranslated"
                )
            for msgid in missing:
                problems.append(f"{language}: no translation for {msgid!r}")
        if check and problems:
            raise CommandError("\n".join(problems))
        if problems:
            self.stdout.write("\n".join(problems))
