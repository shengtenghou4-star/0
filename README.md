# Opaque Task Relay

This repository is a minimal GitHub Actions boundary for opaque requests stored in a dedicated private queue.

## Public contract

- Every public request is identified only by a 32-character lowercase hexadecimal capsule ID.
- Offline task capsules are started manually and execute in a network-disabled container.
- Public HTTPS acquisition capsules may be started manually or by a neutral trigger file containing only the opaque capsule ID.
- The dedicated queue repository and its single-repository credential are fixed in the workflows.
- Public inputs never accept repository names, refs, URLs, shell commands, scientific terms, project names, or output paths.
- URLs and exact host allowlists for public acquisition exist only in the private manifest.
- A networked acquisition container receives no queue credential, private repository name, GitHub environment, host workspace, or arbitrary executable payload.
- Capsule stdout, stderr, outputs, raw public downloads, and receipts are returned only to the private queue.
- No result artifact is uploaded to the public repository.
- Checkout credential persistence is disabled.

Offline capsule code must be authored or reviewed by the laboratory. Public acquisition manifests must also be reviewed. Container isolation and strict contracts reduce accidental credential, workspace, and metadata exposure; they are not a promise of safety against a container-runtime or host-kernel escape.

The public repository is not a source of truth for any project. It contains no project names, scientific inputs, private source code, claims, results, or project-to-repository mapping.
