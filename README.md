# Opaque Task Relay

This repository is a minimal GitHub Actions execution boundary for opaque, pre-staged capsules stored in a private queue.

## Public contract

- The only public input is a 32-character lowercase hexadecimal capsule ID.
- The workflow is manual (`workflow_dispatch`) only.
- The queue repository, queue ref, and credentials are stored exclusively as repository secrets.
- The runner never accepts repository names, refs, URLs, shell commands, or file paths from public inputs.
- Queue credentials are removed from the child-process environment before capsule execution.
- Capsule stdout, stderr, outputs, and receipts are returned only to the private queue.
- No result artifact is uploaded to the public repository.

The public repository is not a source of truth for any project. It contains no project names, scientific inputs, private source code, claims, results, or project-to-repository mapping.
