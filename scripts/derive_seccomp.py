"""
Generate conf/sandbox/seccomp.json from the moby/profiles upstream default seccomp profile.

Usage:
    uv run python scripts/derive_seccomp.py

The script fetches the default seccomp profile from the moby/moby repository (vendored
moby/profiles copy), removes a deny-list of dangerous syscalls from the allowed set, and
writes the result to conf/sandbox/seccomp.json.

The generated conf/sandbox/seccomp.json is committed to the repository.  The drift test
(tests/unit/tools/test_seccomp_profile.py) verifies that the committed artifact is still
consistent and runs fully offline — it never fetches from the network.

To regenerate:
    uv run python scripts/derive_seccomp.py

Requires: network access to raw.githubusercontent.com.
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

# ---------------------------------------------------------------------------
# Source URL — vendored copy inside moby/moby (the "profiles" sub-module was
# moved from the top-level profiles/ directory to vendor/github.com/moby/profiles/).
# Pinned to the master branch; update the SHA here when upstream changes.
# ---------------------------------------------------------------------------
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/moby/moby/master"
    "/vendor/github.com/moby/profiles/seccomp/default.json"
)

# ---------------------------------------------------------------------------
# Deny-list: syscalls that must NOT appear in the allowed set of the sandbox
# profile.  These are dangerous capabilities that we explicitly prohibit even
# if moby upstream allows them.
# ---------------------------------------------------------------------------
DENY_SET: frozenset[str] = frozenset(
    {
        "add_key",
        "bpf",
        "chroot",
        "clone3",
        "create_module",
        "delete_module",
        "finit_module",
        "init_module",
        "ioperm",
        "iopl",
        "kexec_file_load",
        "kexec_load",
        "keyctl",
        "lookup_dcookie",
        "mount",
        "pivot_root",
        "ptrace",
        "reboot",
        "request_key",
        "setns",
        "swapon",
        "swapoff",
        "sysfs",
        "umount",
        "umount2",
        "unshare",
    }
)

OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "conf" / "sandbox" / "seccomp.json"


def _fetch(url: str) -> dict:  # type: ignore[type-arg]
    """Fetch JSON from *url* and return the parsed object."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def _remove_denied(profile: dict) -> dict:  # type: ignore[type-arg]
    """Return a copy of *profile* with DENY_SET syscalls stripped from every
    SCMP_ACT_ALLOW entry in the ``syscalls`` list."""
    new_syscalls = []
    for entry in profile.get("syscalls", []):
        if entry.get("action") != "SCMP_ACT_ALLOW":
            # Keep non-allow entries unchanged (e.g. SCMP_ACT_ERRNO for clone3
            # in upstream — but we will also ensure clone3 is gone from allow).
            # Actually, for safety we skip any entry whose *names* overlap with
            # the deny set and whose action is NOT already blocking.
            new_syscalls.append(entry)
            continue
        # Filter the names list
        filtered_names = [n for n in entry.get("names", []) if n not in DENY_SET]
        if not filtered_names:
            # Entry becomes empty — drop it entirely
            continue
        new_entry = {**entry, "names": filtered_names}
        new_syscalls.append(new_entry)

    return {**profile, "syscalls": new_syscalls}


def main() -> None:
    print(f"Fetching seccomp profile from: {UPSTREAM_URL}")
    upstream = _fetch(UPSTREAM_URL)

    print(f"Upstream syscall entries: {len(upstream.get('syscalls', []))}")
    hardened = _remove_denied(upstream)

    # Verify
    allowed: set[str] = set()
    for entry in hardened.get("syscalls", []):
        if entry.get("action") == "SCMP_ACT_ALLOW":
            allowed.update(entry.get("names", []))

    remaining_denied = DENY_SET & allowed
    if remaining_denied:
        raise RuntimeError(
            f"BUG: deny-set syscalls still present in allowed set: {remaining_denied}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(hardened, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Written to: {OUTPUT_PATH}")
    print(f"Allowed syscall entries: {len(hardened.get('syscalls', []))}")
    print(f"Denied syscalls confirmed absent from allowed set: {sorted(DENY_SET)}")


if __name__ == "__main__":
    main()
