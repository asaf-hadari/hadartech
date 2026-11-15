#!/usr/bin/env python3
"""Drive a Buildroot image on an emulated board through its serial console.

This is the base every task grader builds on: boot the board in QEMU, wait for
the login prompt, log in, run shell commands and read back their output and
exit status. Standard library only, so graders run anywhere Python 3.8+ runs.

Library use:

    from qemu_board import QemuBoard
    with QemuBoard("work/output/aarch64-virt/images") as board:
        board.boot()
        out, rc = board.run("uname -r")

Command line:

    qemu_board.py --images DIR --run "uname -a"      # headless check
    qemu_board.py --images DIR --interactive         # serial console in this terminal
"""

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import uuid

DEFAULT_BOOT_TIMEOUT = 300  # seconds; emulation without KVM is slow
DEFAULT_CMD_TIMEOUT = 60


class BoardError(Exception):
    """The board did not reach the expected state (boot failure, timeout, crash)."""


def qemu_command(images, memory="512", smp="1", extra_append=""):
    """QEMU command line for the aarch64 'virt' base board."""
    kernel = os.path.join(images, "Image")
    rootfs = os.path.join(images, "rootfs.ext4")
    for path in (kernel, rootfs):
        if not os.path.exists(path):
            raise BoardError(f"missing image file: {path} (run 'make base' first)")
    append = "rootwait root=/dev/vda console=ttyAMA0 " + extra_append
    return [
        "qemu-system-aarch64",
        "-M", "virt", "-cpu", "cortex-a53", "-smp", smp, "-m", memory,
        "-nographic", "-no-reboot",
        "-kernel", kernel,
        "-append", append.strip(),
        "-netdev", "user,id=eth0",
        "-device", "virtio-net-device,netdev=eth0",
        # snapshot=on: writes go to a temporary overlay, so every run starts
        # from the same pristine rootfs and a task can never corrupt it.
        "-drive", f"file={rootfs},if=none,format=raw,id=hd0,snapshot=on",
        "-device", "virtio-blk-device,drive=hd0",
    ]


class QemuBoard:
    LOGIN_PROMPT = re.compile(rb"login:\s*$", re.M)
    SHELL_PROMPT = re.compile(rb"# $", re.M)
    PANIC = re.compile(rb"Kernel panic|end Kernel panic|Rebooting in")

    def __init__(self, images, log_path=None, **qemu_opts):
        self.cmd = qemu_command(images, **qemu_opts)
        self.log_path = log_path
        self.proc = None
        self.buffer = b""  # everything read from the console so far
        self._log = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if shutil.which(self.cmd[0]) is None:
            raise BoardError(f"{self.cmd[0]} not found (run 'make deps')")
        if self.log_path:
            self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0)
        return self

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._log:
            self._log.close()
            self._log = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- console I/O -------------------------------------------------------

    def _read_some(self, timeout):
        fd = self.proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return b""
        data = os.read(fd, 4096)
        if not data:
            raise BoardError("QEMU exited (console closed)")
        self.buffer += data
        if self._log:
            self._log.write(data)
            self._log.flush()
        return data

    def expect(self, pattern, timeout, start=0):
        """Wait until `pattern` (bytes regex) matches the console output after
        offset `start`. Returns the match. Raises BoardError on timeout,
        kernel panic or QEMU exit."""
        deadline = time.monotonic() + timeout
        while True:
            m = pattern.search(self.buffer, start)
            if m:
                return m
            if self.PANIC.search(self.buffer, start):
                raise BoardError("kernel panic on console")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BoardError(
                    f"timeout after {timeout}s waiting for {pattern.pattern!r}")
            self._read_some(min(remaining, 1.0))

    def send(self, text):
        self.proc.stdin.write(text.encode())
        self.proc.stdin.flush()

    # -- high level --------------------------------------------------------

    def boot(self, timeout=DEFAULT_BOOT_TIMEOUT, user="root"):
        """Boot to a root shell. Returns seconds taken."""
        t0 = time.monotonic()
        self.expect(self.LOGIN_PROMPT, timeout)
        mark = len(self.buffer)
        self.send(user + "\n")
        self.expect(self.SHELL_PROMPT, 30, start=mark)
        # Quiet kernel messages so they do not interleave with command output.
        self.run("dmesg -n 1", timeout=15)
        return time.monotonic() - t0

    def run(self, command, timeout=DEFAULT_CMD_TIMEOUT):
        """Run a shell command on the board. Returns (output, exit_status)."""
        tag = uuid.uuid4().hex[:8]
        # The marker is split by '' so the echoed command line never matches.
        self.send(f"{command}; echo __END_{tag}''_$?__\n")
        mark = len(self.buffer)
        m = self.expect(
            re.compile(rb"__END_" + tag.encode() + rb"_(\d+)__"), timeout, start=mark)
        raw = self.buffer[mark:m.start()].decode(errors="replace")
        self.expect(self.SHELL_PROMPT, 10, start=m.end())
        # Drop the echoed command line (first line) and normalise line endings.
        lines = raw.replace("\r", "").split("\n")[1:]
        return "\n".join(lines).strip("\n"), int(m.group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", required=True, help="Buildroot output/images directory")
    p.add_argument("--interactive", action="store_true",
                   help="attach the serial console to this terminal (Ctrl-A X quits)")
    p.add_argument("--run", action="append", default=[],
                   help="command to run after boot (repeatable)")
    p.add_argument("--boot-timeout", type=int, default=DEFAULT_BOOT_TIMEOUT)
    p.add_argument("--log", help="write the full console transcript here")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    args = p.parse_args()

    if args.interactive:
        cmd = qemu_command(args.images)
        os.execvp(cmd[0], cmd)

    result = {"booted": False, "boot_seconds": None, "commands": [], "error": None}
    try:
        with QemuBoard(args.images, log_path=args.log) as board:
            result["boot_seconds"] = round(board.boot(timeout=args.boot_timeout), 1)
            result["booted"] = True
            for c in args.run:
                out, rc = board.run(c)
                result["commands"].append({"command": c, "exit": rc, "output": out})
    except BoardError as e:
        result["error"] = str(e)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["booted"]:
            print(f"[ok] booted to a root shell in {result['boot_seconds']}s")
        for c in result["commands"]:
            print(f"$ {c['command']}   (exit {c['exit']})\n{c['output']}")
        if result["error"]:
            print(f"[fail] {result['error']}", file=sys.stderr)
    ok = result["booted"] and not result["error"] and all(
        c["exit"] == 0 for c in result["commands"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
