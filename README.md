# R&D Services

Our R&D services offer end-to-end solutions covering the entire spectrum from design to implementation. We specialize in various platforms and programming languages including:

- Embedded Systems
- PC Applications
- C
- C++
- Java
- Python
- C#
<!-- Add more platforms and languages as needed -->

Our team of experts is dedicated to providing innovative solutions tailored to meet your specific requirements. Whether you need assistance with conceptualization, design, prototyping, or full-scale development, we are here to help.

---

# Embedded Linux Gym

Emulated embedded Linux boards for training and evaluating AI coding agents
(and engineers) on real embedded work: kernel configuration, device trees,
rootfs customisation, cross-compilation and boot debugging. No hardware needed:
every board runs in QEMU, and every task is checked automatically over the
serial console.

## Status

Stage 1 done: the base board builds from source in ~35 min (4 cores, first build)
and boots to a root shell in ~4 s under QEMU without KVM. The harness detects
failing commands and kernel panics, and every run starts from a pristine rootfs.

| Board | Emulated machine | Kernel | Build system |
|---|---|---|---|
| `aarch64-virt` | QEMU `virt`, Cortex-A53 | Linux 6.18.7 | Buildroot 2026.02.3 (LTS) |

## Quick start (Debian/Ubuntu host)

```sh
make deps    # qemu-system-arm, build tools
make base    # download Buildroot, build toolchain + kernel + rootfs (~35 min on 4 cores, first time)
make check   # boot headless, log in over serial, run a command, exit 0 on success
make boot    # interactive serial console (Ctrl-A X to quit)
```

Network access needed by `make base`: `buildroot.org`, `cdn.kernel.org`, and
`sources.buildroot.net` (fallback mirror for every other package source).

## Layout

| Path | Purpose |
|---|---|
| `boards/<board>/defconfig` | Buildroot configuration for each emulated board |
| `harness/qemu_board.py` | Boots a board, drives the serial console, runs commands and returns output + exit status. Base for all task graders. |
| `work/` | Build tree, downloads and images (not committed) |

## How a check works

`harness/qemu_board.py` starts QEMU with the rootfs in snapshot mode (every run
starts from the same pristine image), waits for `login:`, logs in as root, and
runs each command with a unique end marker so it can read back the exact output
and exit status. A kernel panic or timeout fails the check.

```sh
python3 harness/qemu_board.py --images work/output/aarch64-virt/images \
    --run "uname -r" --run "cat /proc/cpuinfo | head -5" --json
```
