# Opaque Task Relay

This repository is a minimal GitHub Actions execution boundary for opaque, pre-staged capsules stored in a dedicated private queue.

## Public contract

- The only public input is a 32-character lowercase hexadecimal capsule ID.
- The workflow is manual (`workflow_dispatch`) only.
- The dedicated queue repository and its single-repository credential are fixed in the workflow.
- The runner never accepts repository names, refs, URLs, shell commands, or file paths from public inputs.
- Capsule code executes inside a network-disabled, read-only Docker container with no queue credential or host workspace mount.
- Capsule stdout, stderr, outputs, and receipts are returned only to the private queue.
- No result artifact is uploaded to the public repository.

Capsules must be authored or reviewed by the laboratory. Container isolation reduces accidental credential and workspace exposure; it is not a promise of safety against kernel-level container escape.

The public repository is not a source of truth for any project. It contains no project names, scientific inputs, private source code, claims, results, or project-to-repository mapping.
