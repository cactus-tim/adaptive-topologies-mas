"""
Drift test for conf/sandbox/seccomp.json.

Runs entirely offline — no network required.  Verifies that the committed
seccomp profile:

1. Is valid JSON and can be loaded.
2. Has defaultAction == "SCMP_ACT_ERRNO" (deny-by-default posture).
3. Contains no syscalls from the deny-set in any SCMP_ACT_ALLOW entry
   (i.e. deny-set ∩ allowed-set == ∅).
"""

from __future__ import annotations

import json
import pathlib

import pytest

SECCOMP_PATH = pathlib.Path(__file__).parent.parent.parent.parent / "conf" / "sandbox" / "seccomp.json"

# Must match DENY_SET in scripts/derive_seccomp.py exactly.
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
        "swapoff",
        "swapon",
        "sysfs",
        "umount",
        "umount2",
        "unshare",
    }
)


@pytest.fixture(scope="module")
def seccomp_profile() -> dict:  # type: ignore[type-arg]
    """Load and return the committed seccomp JSON profile."""
    assert SECCOMP_PATH.exists(), (
        f"conf/sandbox/seccomp.json not found at {SECCOMP_PATH}. "
        "Run: uv run python scripts/derive_seccomp.py"
    )
    with SECCOMP_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def test_seccomp_json_is_valid(seccomp_profile: dict) -> None:  # type: ignore[type-arg]
    """The file must be parseable JSON (fixture handles this implicitly)."""
    assert isinstance(seccomp_profile, dict)


def test_default_action_is_errno(seccomp_profile: dict) -> None:  # type: ignore[type-arg]
    """defaultAction must be SCMP_ACT_ERRNO (deny-by-default)."""
    assert seccomp_profile.get("defaultAction") == "SCMP_ACT_ERRNO", (
        f"Expected defaultAction='SCMP_ACT_ERRNO', "
        f"got {seccomp_profile.get('defaultAction')!r}"
    )


def test_deny_set_not_in_allowed(seccomp_profile: dict) -> None:  # type: ignore[type-arg]
    """No syscall from the deny-set may appear in any SCMP_ACT_ALLOW entry."""
    allowed_syscalls: set[str] = set()
    for entry in seccomp_profile.get("syscalls", []):
        if entry.get("action") == "SCMP_ACT_ALLOW":
            allowed_syscalls.update(entry.get("names", []))

    intersection = DENY_SET & allowed_syscalls
    assert not intersection, (
        f"Denied syscall(s) found in allowed set — regenerate seccomp.json:\n"
        f"  Run: uv run python scripts/derive_seccomp.py\n"
        f"  Offenders: {sorted(intersection)}"
    )


def test_syscalls_list_is_non_empty(seccomp_profile: dict) -> None:  # type: ignore[type-arg]
    """The profile must have at least one syscall entry."""
    assert len(seccomp_profile.get("syscalls", [])) > 0
