# Texts and settings of a public server, by domain

A server set up with [deploy/server](../server/README.md) reads the folder of its domain,
`deploy/sites/<domain>/`, where `<domain>` is `BRICKDNN_DOMAIN` in the server's `.env`:

- `legal/imprint.md`, `legal/privacy.md`, `legal/terms.md`: the imprint, privacy policy and
  terms of use (templates and a checklist in [deploy/legal](../legal/README.md)). Files in
  `/srv/brick-dnn/legal/` on the server take precedence.
- `settings.env`: `BRICKDNN_…=value` lines, defaults for the backend, the training worker and
  the cleanup service, e.g. the plans and the periods of the automatic deletion. A value set
  in the server's `.env` wins. Never put secrets here: the repository is public.

Both go online with the next deploy, like the code. Only the server of that domain uses its
folder, so a fork deployed elsewhere does not publish these texts or use these settings.
