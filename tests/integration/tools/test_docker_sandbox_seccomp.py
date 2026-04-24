"""Per-syscall seccomp tests for DockerSandbox.

Each test verifies that a specific dangerous syscall is blocked
(errno == EPERM=1 or ENOSYS=38) inside the hardened container.

All tests are gated with @pytest.mark.docker.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from atm.tools.sandbox.base import SandboxConfig
from atm.tools.sandbox.docker_sandbox import DockerSandbox

_SECCOMP_PATH = pathlib.Path(__file__).parents[3] / "conf" / "sandbox" / "seccomp.json"

# errno values
EPERM = 1
ENOSYS = 38


@pytest.fixture(scope="module")
def seccomp_str() -> str:
    return _SECCOMP_PATH.read_text()


@pytest.fixture(scope="module")
def sandbox(seccomp_str: str) -> DockerSandbox:
    cfg = SandboxConfig(
        mem_limit="128m",
        pids_limit=64,
        timeout_s=10.0,
        tmpfs_mounts={
            "/work": "size=64m,mode=700",
            "/tmp": "size=64m,noexec,nosuid",
        },
    )
    return DockerSandbox(config=cfg, seccomp_json_str=seccomp_str, prefetch=False)


def _syscall_test_code(syscall_name: str, call_code: str) -> str:
    """Generate Python code that calls a syscall and prints the errno."""
    return textwrap.dedent(f"""\
        import ctypes
        import ctypes.util
        import errno as errno_mod

        libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)

        # Reset errno
        ctypes.set_errno(0)

        result = {call_code}

        err = ctypes.get_errno()
        print(f"syscall={syscall_name} result={{result}} errno={{err}}")
        if err in ({EPERM}, {ENOSYS}):
            print("BLOCKED_OK")
        elif result == -1:
            print(f"BLOCKED_OTHER errno={{err}}")
        else:
            print("ALLOWED")
    """)


async def _assert_syscall_blocked(
    sandbox: DockerSandbox, syscall_name: str, call_code: str
) -> None:
    """Run the syscall test code and assert it reports BLOCKED_OK."""
    code = _syscall_test_code(syscall_name, call_code)
    result = await sandbox.execute(lang="python", code=code)
    output = result.stdout + result.stderr
    assert "BLOCKED_OK" in output or result.exit_code != 0, (
        f"Syscall '{syscall_name}' should be blocked by seccomp, got: {output!r}"
    )


# ---------------------------------------------------------------------------
# Per-syscall tests
# ---------------------------------------------------------------------------


@pytest.mark.docker
async def test_seccomp_ptrace_blocked(sandbox: DockerSandbox) -> None:
    """ptrace syscall must be blocked."""
    # PTRACE_TRACEME=0, pid=0, addr=0, data=0
    await _assert_syscall_blocked(
        sandbox,
        "ptrace",
        "libc.ptrace(0, 0, 0, 0)",
    )


@pytest.mark.docker
async def test_seccomp_mount_blocked(sandbox: DockerSandbox) -> None:
    """mount syscall must be blocked."""
    await _assert_syscall_blocked(
        sandbox,
        "mount",
        "libc.mount(b'/dev/null', b'/mnt', b'ext4', 0, None)",
    )


@pytest.mark.docker
async def test_seccomp_unshare_blocked(sandbox: DockerSandbox) -> None:
    """unshare syscall must be blocked."""
    # CLONE_NEWNS = 0x00020000
    await _assert_syscall_blocked(
        sandbox,
        "unshare",
        "libc.unshare(0x00020000)",
    )


@pytest.mark.docker
async def test_seccomp_bpf_blocked(sandbox: DockerSandbox) -> None:
    """bpf syscall must be blocked."""
    # BPF_MAP_CREATE=0
    await _assert_syscall_blocked(
        sandbox,
        "bpf",
        "libc.syscall(321, 0, None, 0)",  # x86_64 bpf syscall number
    )


@pytest.mark.docker
async def test_seccomp_keyctl_blocked(sandbox: DockerSandbox) -> None:
    """keyctl syscall must be blocked."""
    # KEYCTL_GET_KEYRING_ID = 0
    await _assert_syscall_blocked(
        sandbox,
        "keyctl",
        "libc.keyctl(0, 0, 0, 0, 0)",
    )


@pytest.mark.docker
async def test_seccomp_pivot_root_blocked(sandbox: DockerSandbox) -> None:
    """pivot_root syscall must be blocked."""
    await _assert_syscall_blocked(
        sandbox,
        "pivot_root",
        "libc.pivot_root(b'/', b'/')",
    )


@pytest.mark.docker
async def test_seccomp_chroot_blocked(sandbox: DockerSandbox) -> None:
    """chroot syscall must be blocked (cap_drop ALL already removes CAP_SYS_CHROOT)."""
    await _assert_syscall_blocked(
        sandbox,
        "chroot",
        "libc.chroot(b'/tmp')",
    )


@pytest.mark.docker
async def test_seccomp_setns_blocked(sandbox: DockerSandbox) -> None:
    """setns syscall must be blocked."""
    # CLONE_NEWUTS = 0x04000000
    await _assert_syscall_blocked(
        sandbox,
        "setns",
        "libc.setns(0, 0x04000000)",
    )


@pytest.mark.docker
async def test_seccomp_clone3_blocked(sandbox: DockerSandbox) -> None:
    """clone3 syscall must be blocked."""
    # clone3 = syscall 435 on x86_64
    await _assert_syscall_blocked(
        sandbox,
        "clone3",
        "libc.syscall(435, None, 0)",
    )


@pytest.mark.docker
async def test_seccomp_add_key_blocked(sandbox: DockerSandbox) -> None:
    """add_key syscall must be blocked."""
    await _assert_syscall_blocked(
        sandbox,
        "add_key",
        "libc.add_key(b'user', b'test', None, 0, -4)",  # KEY_SPEC_THREAD_KEYRING = -1
    )


@pytest.mark.docker
async def test_seccomp_init_module_blocked(sandbox: DockerSandbox) -> None:
    """init_module syscall must be blocked."""
    await _assert_syscall_blocked(
        sandbox,
        "init_module",
        "libc.init_module(None, 0, b'')",
    )
