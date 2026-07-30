# Operational notes

This repository is a public compatibility harness with a deliberately narrow interface.

- Runs accept only fixed-format case identifiers.
- Jobs use fixed implementations rather than arbitrary user-supplied commands.
- Secrets, repository names, source locations, URLs, case meanings, and private records must not be placed in public inputs, logs, documentation, branch names, pull requests, or commit messages.
- Generated materials are not retained as public artifacts.
- Checkout credential persistence is disabled.

The repository is not an authoritative data source. Its public surface is limited to generic compatibility machinery and unrelated public dependencies required by fixed jobs.
