# Texts of a public server, by domain

A server set up with [deploy/server](../server/README.md) shows the imprint, privacy policy and
terms of use from `deploy/sites/<domain>/legal/` (`imprint.md`, `privacy.md`, `terms.md`),
where `<domain>` is `BRICKDNN_DOMAIN` in the server's `.env`. They go online with the next
deploy, like the code. Files in `/srv/brick-dnn/legal/` on the server take precedence.

Only the server of that domain uses its folder, so a fork deployed elsewhere does not publish
these texts. Templates and a checklist: [deploy/legal](../legal/README.md).
