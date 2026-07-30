# Security model

## Trust boundaries

The public repository is trusted only as a fixed launcher. The private queue is the authority for capsule payloads and integrity metadata.

## Required secrets

- `LAB_QUEUE_REPOSITORY`: private queue in `owner/repository` form.
- `LAB_QUEUE_REF`: immutable or tightly controlled queue ref used for request reads.
- `LAB_QUEUE_READ_TOKEN`: fine-grained token with read-only Contents access to the queue.
- `LAB_QUEUE_WRITE_BRANCH`: existing private queue branch used for result commits.
- `LAB_QUEUE_WRITE_TOKEN`: separate fine-grained token with Contents write access to the queue.

Tokens must be scoped to the single private queue repository. They must not have administration, Actions, Issues, Pull requests, Packages, or organization permissions.

## Prohibited public data

Public inputs, workflow names, branch names, pull requests, commit messages, logs, and README files must not contain project names, private repository numbers, scientific terms, candidate identifiers, experiment names, result summaries, or research progress.

## Capsule contract

The queue stores:

`relay/requests/<capsule_id>/manifest.json`

`relay/requests/<capsule_id>/payload.tar.gz.b64`

The payload file is Base64 text wrapping a gzip tar archive. The manifest is fail-closed. The payload must contain a root-level executable `run.sh`. Archive links, devices, absolute paths, and parent traversal are rejected. The payload SHA-256 must match the manifest.

The runner passes only a minimal environment to `run.sh`, captures output without streaming it to public logs, and returns a compressed result package to a run-specific private path.
