# Security model

## Trust boundaries

The public repository is trusted only as a fixed launcher. The dedicated private queue is the authority for capsule payloads, public-fetch manifests, and integrity metadata.

The queue credential must be a fine-grained token scoped only to `shengtenghou4-star/00`, with repository Contents read and write permission and no other repository selected.

Checkout credential persistence is disabled. The queue token is used only by the host orchestrator to read requests and return private results.

## Public information boundary

Public inputs, workflow and branch names, pull requests, commit messages, logs, trigger files, and documentation must not disclose private project names, private repository numbers other than the neutral queue `00`, private data, candidate identifiers, unpublished experiment names, research results or progress, credentials, private destination paths, or any mapping from an opaque task to a private project.

A fixed audited workflow may identify a generic public dependency, an immutable public-source commit, or a public dataset/provider/DOI when that identity is required for reproducibility. These public identities must not be coupled to a private repository, private project, candidate, result, or progress claim. Dynamic source URLs, exact acquisition allowlists, private task manifests, and project-to-task mappings remain private.

Historical migration and diagnostic commits may name a public dependency or public dataset source. They are operational receipts only and are not research authority or evidence of a private project mapping.

## Offline capsule contract

The queue stores:

`relay/requests/<capsule_id>/manifest.json`

`relay/requests/<capsule_id>/payload.tar.gz.b64`

The payload file is Base64 text wrapping a gzip tar archive. The manifest is fail-closed. The payload must contain a root-level executable `run.sh`. Archive links, devices, absolute paths, and parent traversal are rejected. The payload SHA-256 must match the manifest.

Offline capsule code runs in a separate Docker PID and mount namespace. The container:

- receives no queue token, repository name, ref, branch, or GitHub environment;
- has no network;
- has a read-only root filesystem;
- has all Linux capabilities dropped;
- uses `no-new-privileges`;
- mounts the capsule read-only and only the result directory writable;
- does not receive the host workspace or Docker socket.

## Public HTTPS acquisition contract

The queue stores:

`relay/public-fetch/<capsule_id>/manifest.json`

The strict manifest may contain only:

- the matching opaque capsule ID;
- the frozen `public_https_fetch` operation;
- a timeout and global byte cap;
- exact lowercase HTTPS host allowlists;
- an optional landing URL;
- one to eight safe output names, candidate HTTPS URLs, byte bounds, and optional magic bytes.

No shell command, repository name, token, executable payload, public path, or arbitrary environment value is accepted.

The acquisition worker runs in a separate networked Docker container. The container:

- receives no queue token, private repository name, ref, branch, or GitHub environment;
- receives no arbitrary executable capsule;
- validates every initial and redirected URL against the exact HTTPS host allowlist;
- rejects hosts resolving to non-global IP addresses at validation time;
- has a read-only root filesystem;
- has all Linux capabilities dropped;
- uses `no-new-privileges`;
- mounts only the validated manifest and fixed public worker read-only, with a result directory writable;
- does not receive the host workspace or Docker socket.

Downloads are bounded by per-file and global byte limits and may be required to match frozen magic bytes. The result package is split into bounded private chunks and returned to the private queue.

## Output handling

Both orchestrators capture worker stdout and stderr without streaming private details to public logs. Public logs receive only generic success or fail-closed annotations. Result packages and receipts are returned only to the private queue.

Offline capsules must be authored or reviewed by the laboratory. Public acquisition manifests must also be reviewed. The design does not claim resistance to DNS infrastructure compromise, container-runtime escape, or host-kernel escape.
