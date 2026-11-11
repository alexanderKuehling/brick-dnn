"""``python manage.py send_test_email <address>``: check the e-mail settings (docs/email.md).

Prints the settings in use (never the password), connects to the SMTP server and sends a short
e-mail in the site's layout. In Docker: ``docker compose exec backend python manage.py
send_test_email you@example.com``.
"""

from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.core.validators import validate_email

from brick_dnn_core import mail, notices


class Command(BaseCommand):
    help = "Send a test e-mail to ADDRESS with the e-mail settings of this server."

    def add_arguments(self, parser) -> None:
        parser.add_argument("address", help="Where to send the test e-mail.")

    def handle(self, *args, **options) -> None:
        address = options["address"].strip()
        try:
            validate_email(address)
        except ValidationError:
            raise CommandError(f"Not an e-mail address: {address!r}") from None
        if not mail.enabled():
            raise CommandError(
                "E-mail is not set up: set BRICKDNN_EMAIL_HOST (and the other BRICKDNN_EMAIL_… "
                "variables, see docs/email.md), then restart the backend."
            )
        backend = settings.EMAIL_BACKEND.rsplit(".", 2)[-2]
        security = (
            "SSL/TLS"
            if settings.EMAIL_USE_SSL
            else "STARTTLS"
            if settings.EMAIL_USE_TLS
            else "none"
        )
        self.stdout.write(f"Backend:  {backend}")
        if backend == "smtp":
            self.stdout.write(f"Server:   {settings.EMAIL_HOST}:{settings.EMAIL_PORT} ({security})")
            self.stdout.write(f"Login:    {settings.EMAIL_HOST_USER or '(none)'}")
            self.stdout.write(f"Password: {'set' if settings.EMAIL_HOST_PASSWORD else '(none)'}")
        self.stdout.write(f"From:     {settings.DEFAULT_FROM_EMAIL}")
        if settings.EMAIL_REPLY_TO:
            self.stdout.write(f"Reply-To: {', '.join(settings.EMAIL_REPLY_TO)}")
        links = mail.site_url() or "(not set: set BRICKDNN_PUBLIC_URL)"
        self.stdout.write(f"Links:    {links}")
        if "@localhost" in settings.DEFAULT_FROM_EMAIL:
            self.stdout.write(
                self.style.WARNING(
                    "BRICKDNN_EMAIL_FROM is not set: providers reject mails from @localhost."
                )
            )
        try:
            mail.check_connection()
        except Exception as exc:
            raise CommandError(f"Cannot connect to the e-mail server: {exc}") from None
        if not notices.test_email(address):
            raise CommandError("The e-mail was not sent; see the log above.")
        self.stdout.write(self.style.SUCCESS(f"Sent a test e-mail to {address}."))
