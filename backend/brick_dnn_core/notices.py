"""The e-mails Brick DNN sends to its users (sent with mail.py, only if e-mail is set up).

- links: verifying an e-mail address, resetting the password (recovery.py);
- security notices, short and only for what an account holder should know about: the password
  was changed, the e-mail address was changed (to the old address), two-factor authentication
  was turned on or off, an API token was created or an app was connected;
- the warning before an inactive account is deleted (retention.py).

Security notices go only to an address that may receive password reset links
(recovery.trusted_email): not to an address someone typed and never confirmed. Deletion warnings
go to any address of the account: they reveal nothing and keep accounts from being lost.

Every e-mail is written in the recipient's language (languages.user_language: the one saved on
the account, else the request's, else English); the texts are in ``locale/de``.
"""

from __future__ import annotations

import functools
from datetime import date

from django.conf import settings
from django.utils import translation
from django.utils.formats import date_format
from django.utils.translation import gettext as _

from . import languages, mail, recovery


def _in_language_of_user(send):
    """Run ``send(user, ...)`` in the language of its user (and request, if given)."""

    @functools.wraps(send)
    def wrapper(user, *args, **kwargs):
        request = kwargs.get("request")
        if request is None:
            request = next((arg for arg in args if hasattr(arg, "META")), None)
        with translation.override(languages.user_language(user, request)):
            return send(user, *args, **kwargs)

    return wrapper


def _reason(user, request=None) -> str:
    base = mail.site_url(request)
    where = mail.site_domain(base) if base else settings.SITE_NAME
    return _("You get this e-mail because of your account “%(username)s” at %(site)s.") % {
        "username": user.get_username(),
        "site": where,
    }


def _greeting(user) -> str:
    return _("Hello %(username)s,") % {"username": user.get_username()}


def _account_link(request=None) -> tuple[str, str]:
    return (_("Open your account"), mail.link("/account", request))


def _if_not_you() -> str:
    return _(
        "If this was not you, change your password right away and turn on two-factor "
        "authentication on your account page, or ask the operator of the site for help."
    )


def _notice(user, subject: str, paragraphs: list[str], request=None, to: str | None = None):
    address = to if to is not None else recovery.trusted_email(user)
    if not address:
        return False
    return mail.send(
        mail.Message(
            to=address,
            subject=subject,
            greeting=_greeting(user),
            paragraphs=paragraphs,
            action=_account_link(request),
            after=[_if_not_you()],
            reason=_reason(user, request),
            base_url=mail.site_url(request),
        )
    )


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


@_in_language_of_user
def verify_email(user, url: str, request=None) -> bool:
    return mail.send(
        mail.Message(
            to=user.email,
            subject=_("Confirm your e-mail address"),
            greeting=_greeting(user),
            paragraphs=[
                _(
                    "Please confirm that %(email)s is the e-mail address of your account. "
                    "Open the link and log in if you are asked to."
                )
                % {"email": user.email},
            ],
            action=(_("Confirm my e-mail address"), url),
            after=[
                _(
                    "The link is valid for %(days)d days. If you did not sign up or change your "
                    "address, ignore this e-mail: nothing changes without the link."
                )
                % {"days": recovery.VERIFY_DAYS},
            ],
            reason=_reason(user, request),
            base_url=mail.site_url(request),
        )
    )


@_in_language_of_user
def password_reset(user, url: str, request=None) -> bool:
    return mail.send(
        mail.Message(
            to=user.email,
            subject=_("Reset your password"),
            greeting=_greeting(user),
            paragraphs=[
                _(
                    "Someone asked to reset the password of your account. If it was you, choose "
                    "a new password with this link. It is valid for one hour and works once."
                ),
            ],
            action=(_("Choose a new password"), url),
            after=[
                _(
                    "If you did not ask for this, ignore this e-mail: your password stays as it "
                    "is. Two-factor authentication, if you turned it on, stays on after a reset."
                ),
            ],
            reason=_reason(user, request),
            base_url=mail.site_url(request),
        )
    )


# --------------------------------------------------------------------------- #
# Security notices
# --------------------------------------------------------------------------- #


@_in_language_of_user
def password_changed(user, request=None) -> bool:
    return _notice(
        user,
        _("Your password was changed"),
        [_("The password of your account was changed. All other logins were ended.")],
        request,
    )


@_in_language_of_user
def email_changed(user, old_address: str, request=None) -> bool:
    return _notice(
        user,
        _("Your e-mail address was changed"),
        [
            _(
                "The e-mail address of your account was changed from %(old)s to %(new)s. "
                "Further e-mails go to the new address."
            )
            % {"old": old_address, "new": user.email},
        ],
        request,
        to=old_address,
    )


@_in_language_of_user
def two_factor_changed(user, enabled: bool, request=None) -> bool:
    if enabled:
        subject = _("Two-factor authentication turned on")
        text = _(
            "Two-factor authentication is now on for your account: logging in needs a code "
            "from your authenticator app. Keep your recovery codes in a safe place."
        )
    else:
        subject = _("Two-factor authentication turned off")
        text = _(
            "Two-factor authentication was turned off for your account: logging in needs "
            "only your password now."
        )
    return _notice(user, subject, [text], request)


@_in_language_of_user
def api_token_created(user, name: str, request=None) -> bool:
    return _notice(
        user,
        _("A new API token was created"),
        [
            _(
                "An API token named “%(name)s” was created for your account. Scripts and AI "
                "assistants can use it to work with your networks and data. You can delete it "
                "on the page API tokens."
            )
            % {"name": name},
        ],
        request,
    )


@_in_language_of_user
def app_connected(user, app: str, request=None) -> bool:
    return _notice(
        user,
        _("An app was connected to your account"),
        [
            _(
                "The app “%(app)s” was connected to your account (OAuth) and can now use your "
                "networks and data through the MCP server. You can disconnect it on the page "
                "API tokens."
            )
            % {"app": app},
        ],
        request,
    )


# --------------------------------------------------------------------------- #
# Inactive accounts
# --------------------------------------------------------------------------- #


@_in_language_of_user
def deletion_warning(user, delete_on: date, inactive_days: int) -> bool:
    """Sent right away (the purge needs to know whether it went out)."""
    return mail.send(
        mail.Message(
            to=user.email,
            subject=_("Your account will be deleted"),
            greeting=_greeting(user),
            paragraphs=[
                _(
                    "Nobody has used your account for a long time. Accounts that are not used "
                    "for %(days)d days are deleted with everything in them: your networks, data "
                    "and trainings. Yours will be deleted on or after %(date)s."
                )
                % {
                    "days": inactive_days,
                    # Django's date format: "15 November 2026", "15. November 2026".
                    "date": date_format(delete_on, _("j F Y")),
                },
                _(
                    "To keep it, simply log in before then. If you do not need it any more, you "
                    "do not have to do anything; you can also download your data on the account "
                    "page first."
                ),
            ],
            action=(_("Log in to keep my account"), mail.link("/login")),
            reason=_reason(user),
        ),
        background=False,
    )


def test_email(address: str) -> bool:
    return mail.send(
        mail.Message(
            to=address,
            subject="Test e-mail",
            paragraphs=[
                "This is a test e-mail of the server. If you can read it, sending e-mail works.",
            ],
            reason="Sent with `manage.py send_test_email`.",
        ),
        background=False,
    )
