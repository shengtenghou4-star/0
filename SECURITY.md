# Security model

## Trust boundaries

The public repository is trusted only as a fixed launcher. The dedicated private queue is the authority for capsule payloads and integrity metadata.

The queue credential must be a fine-grained token scoped only to `shengtenghou4-star/00`, with repository Contents read and write permission and no other repository selected.

## Prohibited public data

Public inputs, workflow names, branch names, pull requests, commit messages, logs, and README files must not contain project names, private repository numbers other than the neutral queue `00`, scientific terms, candidate identifiers, experiment names, result summaries, or research progress.

## Capsule contract

The queue stores:

`relay/requests/<capsule_id>/manifest.json`

`relay/requests/<capsule_id>/payload.tar.gz.b64`

The payload file is Base64 text wrapping a gzip tar archive. The manifest is fail-closed. The payload must contain a root-level executable `run.sh`. Archive links, devices, absolute paths, and parent traversal are rejected. The payload SHA-256 must match the manifest.

## Execution containment

Capsule code runs in a separate Docker PID and mount namespace. The container:

- receives no queue token, repository name, ref, branch, or GitHub environment;
- has no network;
- has a read-only root filesystem;
- has all Linux capabilities dropped;
- uses `no-new-privileges`;
- mounts the capsule read-only and only the result directory writable;
- does not receive the host workspace or Docker socket.

The host launcher captures capsule output without streaming it to public logs and returns a compressed result package to the private queue after the container exits.

Capsules must still be authored or reviewed by the laboratory. This design is not intended to execute adversarial code capable of exploiting the container runtime or host kernel.
