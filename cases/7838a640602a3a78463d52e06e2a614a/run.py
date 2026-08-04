from __future__ import annotations

import json
import os
import platform
import urllib.error
import urllib.request
from pathlib import Path

CASE_ID = "7838a640602a3a78463d52e06e2a614a"
TOKEN_NAMES = (
    "LAZARUS_REPO_TOKEN",
    "PRIVATE_REPO_TOKEN",
    "GH_PAT",
    "REPO_TOKEN",
)


def probe_private_repo(token: str) -> tuple[bool, int | None, str]:
    request = urllib.request.Request(
        "https://api.github.com/repos/shengtenghou4-star/19",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": f"fixed-case-{CASE_ID}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return (
                response.status == 200 and payload.get("full_name") == "shengtenghou4-star/19",
                response.status,
                "repository-visible" if payload.get("private") is True else "unexpected-payload",
            )
    except urllib.error.HTTPError as exc:
        return False, exc.code, "http-error"
    except Exception as exc:  # diagnostic receipt only
        return False, None, type(exc).__name__


def main() -> None:
    artifact_dir = Path(__file__).parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    token_presence = {name: bool(os.environ.get(name, "")) for name in TOKEN_NAMES}
    private_access = False
    selected_token_name: str | None = None
    http_status: int | None = None
    probe_note = "no-token-present"
    for name in TOKEN_NAMES:
        token = os.environ.get(name, "")
        if not token:
            continue
        ok, status, note = probe_private_repo(token)
        if ok:
            private_access = True
            selected_token_name = name
            http_status = status
            probe_note = note
            break
        if http_status is None:
            http_status = status
            probe_note = note

    result = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "runner_started": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "token_presence": token_presence,
        "private_repository_access": private_access,
        "selected_token_name": selected_token_name,
        "private_probe_http_status": http_status,
        "private_probe_note": probe_note,
        "private_bytes_read": 0,
        "model_fits": 0,
        "confirmation_targets_read": 0,
    }
    (artifact_dir / "probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
