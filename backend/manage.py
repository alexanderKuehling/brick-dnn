#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys


def main() -> None:
    from brick_dnn_site.env_file import load_env_file

    load_env_file()  # settings from ../.env, if the file exists
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "brick_dnn_site.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
