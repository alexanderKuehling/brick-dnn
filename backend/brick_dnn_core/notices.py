"""The e-mails Brick DNN sends to its users (sent with mail.py, only if e-mail is set up).

- links: verifying an e-mail address, resetting the password (recovery.py);
- security notices, short and only for what an account holder should know about: the password
  was changed, the e-mail address was changed (to the old address), two-factor authentication
  was turned on or off, an API token was created or an app was connected;
- the warning before an inactive account is deleted (retention.py).

Security notices go only to an address that may receive password reset links
(recovery.trusted_email): not to an address someone typed and never confirmed. Deletion warnings
go to any address of the account: they reveal nothing and keep accounts from being lost.
"""

from __future__ import annotations

from datetime import date

from django.conf import settings

from . import mail, recovery


def _reason(user, request=None) -> str:
    base = mail.site_url(request)
    where = mail.site_domain(base) if base else settings.SITE_NAME
    return f"You get this e-mail because of your account “{user.get_username()}” at {where}."


def _greeting(user) -> str:
    return f"Hello {user.get_username()},"


def _account_link(request=None) -> tuple[str, str]:
    return ("Open your account", mail.link("/account", request))


IF_NOT_YOU = (
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
            after=[IF_NOT_YOU],
            reason=_reason(user, request),
            base_url=mail.site_url(request),
        )
    )


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


def verify_email(user, url: str, request=None) -> bool:
    return mail.send(
        mail.Message(
            to=user.email,
            subject="Confirm your e-mail address",
            greeting=_greeting(user),
            paragraphs=[
                f"Please confirm that {user.email} is the e-mail address of your account. "
                "Open the link and log in if you are asked to.",
            ],
            action=("Confirm my e-mail address", url),
            after=[
                f"The link is valid for {recovery.VERIFY_DAYS} days. If you did not sign up or "
                "change your address, ignore this e-mail: nothing changes without the link.",
            ],
            reason=_reason(user, request),
            base_url=mail.site_url(request),
        )
    )


def password_reset(user, url: str, request=None) -> bool:
    return mail.send(
        mail.Message(
            to=user.email,
            subject="Reset your password",
            greeting=_greeting(user),
            paragraphs=[
                "Someone asked to reset the password of your account. If it was you, choose "
                "a new password with this link. It is valid for one hour and works once.",
            ],
            action=("Choose a new password", url),
            after=[
                "If you did not ask for this, ignore this e-mail: your password stays as it is. "
                "Two-factor authentication, if you turned it on, stays on after a reset.",
            ],
            reason=_reason(user, request),
            base_url=mail.site_url(request),
        )
    )


# --------------------------------------------------------------------------- #
# Security notices
# --------------------------------------------------------------------------- #


def password_changed(user, request=None) -> bool:
    return _notice(
        user,
        "Your password was changed",
        ["The password of your account was changed. All other logins were ended."],
        request,
    )


def email_changed(user, old_address: str, request=None) -> bool:
    return _notice(
        user,
        "Your e-mail address was changed",
        [
            f"The e-mail address of your account was changed from {old_address} to "
            f"{user.email}. Further e-mails go to the new address.",
        ],
        request,
        to=old_address,
    )


def two_factor_changed(user, enabled: bool, request=None) -> bool:
    if enabled:
        text = (
            "Two-factor authentication is now on for your account: logging in needs a code "
            "from your authenticator app. Keep your recovery codes in a safe place."
        )
    else:
        text = (
            "Two-factor authentication was turned off for your account: logging in needs "
            "only your password now."
        )
    subject = f"Two-factor authentication turned {'on' if enabled else 'off'}"
    return _notice(user, subject, [text], request)


def api_token_created(user, name: str, request=None) -> bool:
    return _notice(
        user,
        "A new API token was created",
        [
            f"An API token named “{name}” was created for your account. Scripts and AI "
            "assistants can use it to work with your networks and data. You can delete it on "
            "the page API tokens.",
        ],
        request,
    )


def app_connected(user, app: str, request=None) -> bool:
    return _notice(
        user,
        "An app was connected to your account",
        [
            f"The app “{app}” was connected to your account (OAuth) and can now use your "
            "networks and data through the MCP server. You can disconnect it on the page "
            "API tokens.",
        ],
        request,
    )


# --------------------------------------------------------------------------- #
# Inactive accounts
# --------------------------------------------------------------------------- #


def deletion_warning(user, delete_on: date, inactive_days: int) -> bool:
    """Sent right away (the purge needs to know whether it went out)."""
    return mail.send(
        mail.Message(
            to=user.email,
            subject="Your account will be deleted",
            greeting=_greeting(user),
            paragraphs=[
                f"Nobody has used your account for a long time. Accounts that are not used "
                f"for {inactive_days} days are deleted with everything in them: your "
                f"networks, data and trainings. Yours will be deleted on or after "
                f"{delete_on.strftime('%d %B %Y')}.",
                "To keep it, simply log in before then. If you do not need it any more, you do "
                "not have to do anything; you can also download your data on the account page "
                "first.",
            ],
            action=("Log in to keep my account", mail.link("/login")),
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
