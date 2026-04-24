# conf/sandbox/seccomp.json — Sandbox Seccomp Profile

## Overview

`seccomp.json` is the seccomp (Secure Computing Mode) profile applied to every
Docker container spawned by `DockerSandbox`.  It is derived from the **moby/profiles**
upstream default profile using a *deny-list* approach: we start from the moby default
(which already allows only a curated set of syscalls) and additionally remove a set of
dangerous syscalls that must never be available inside the sandbox.

The file is **committed to the repository** as a generated artifact.  CI environments
use the committed JSON directly — no network access is required at test or build time.

---

## How to Regenerate

Regeneration requires network access to `raw.githubusercontent.com`.  Run once from
the repository root:

```bash
uv run python scripts/derive_seccomp.py
```

This will:

1. Fetch the upstream profile from:
   `https://raw.githubusercontent.com/moby/moby/master/vendor/github.com/moby/profiles/seccomp/default.json`
2. Remove every syscall in the deny-list (see below) from the `SCMP_ACT_ALLOW` entries.
3. Write the result to `conf/sandbox/seccomp.json` (pretty-printed, `indent=2`,
   `sort_keys=True`).

After regeneration, commit both `conf/sandbox/seccomp.json` and any changes to
`scripts/derive_seccomp.py`.

---

## Source and Version Pinning

| Item | Value |
|------|-------|
| Upstream repo | `moby/moby` |
| Upstream branch | `master` |
| Upstream file | `vendor/github.com/moby/profiles/seccomp/default.json` |
| Derive script | `scripts/derive_seccomp.py` |
| Generated artifact | `conf/sandbox/seccomp.json` |

To pin to a specific commit rather than `master`, update `UPSTREAM_URL` in
`scripts/derive_seccomp.py` to use a full commit SHA:

```
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/moby/moby/<SHA>"
    "/vendor/github.com/moby/profiles/seccomp/default.json"
)
```

---

## Deny-List

The following syscalls are explicitly removed from the allowed set regardless of what
the upstream profile permits.  They represent dangerous kernel interfaces that could be
used to escape the container, modify the host kernel, or gain elevated privileges.

| Syscall | Reason |
|---------|--------|
| `ptrace` | Process tracing / memory inspection of host processes |
| `mount` | Mount filesystems — could expose host paths |
| `umount` | Unmount filesystems |
| `umount2` | Unmount with flags |
| `unshare` | Create new namespaces — partial container escape vector |
| `keyctl` | Kernel keyring — could access host credentials |
| `bpf` | eBPF programs — kernel code execution |
| `pivot_root` | Change root filesystem |
| `chroot` | Change apparent root directory |
| `setns` | Join existing namespaces — container escape |
| `clone3` | Create processes/threads with extended flags |
| `add_key` | Add key to kernel keyring |
| `request_key` | Search kernel keyring |
| `finit_module` | Load kernel module from fd |
| `init_module` | Load kernel module |
| `delete_module` | Remove kernel module |
| `create_module` | Create loadable module entry (legacy) |
| `ioperm` | Set I/O port permissions |
| `iopl` | Change I/O privilege level |
| `kexec_load` | Load new kernel for execution |
| `kexec_file_load` | Load new kernel for execution (file-based) |
| `reboot` | Reboot or halt the system |
| `swapon` | Enable swap area |
| `swapoff` | Disable swap area |
| `sysfs` | Get filesystem type information (deprecated) |
| `lookup_dcookie` | Return directory entry from dcache |

---

## Drift Test

`tests/unit/tools/test_seccomp_profile.py` runs entirely offline and verifies:

1. `seccomp.json` is valid JSON and parseable.
2. `defaultAction == "SCMP_ACT_ERRNO"` (deny-by-default posture).
3. The intersection of the deny-list and the allowed syscall names is empty — i.e.,
   no denied syscall appears in any `SCMP_ACT_ALLOW` entry.

Run with:

```bash
uv run pytest tests/unit/tools/test_seccomp_profile.py -v
```

---

## Usage in DockerSandbox

`DockerSandbox` loads the JSON file once at construction time and passes the profile
content as a JSON **string** via the `security_opt` parameter:

```python
security_opt=["no-new-privileges", f"seccomp={seccomp_json_string}"]
```

This is required because docker-py does not expose a file-path interface for seccomp
profiles — the full JSON content must be embedded in the option string.
