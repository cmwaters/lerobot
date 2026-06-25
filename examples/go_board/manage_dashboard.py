#!/usr/bin/env python
"""Start and stop the Go-board dashboard on laptop or desktop profiles."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GO_BOARD_DIR = Path("examples/go_board")
LAPTOP_CONFIG = GO_BOARD_DIR / "dashboard_config.laptop.json"
DESKTOP_CONFIG = GO_BOARD_DIR / "dashboard_config.desktop.json"
LOCAL_SESSION = "go-dashboard"
LOCAL_LOG = Path("outputs/dashboard_logs/dashboard_laptop_recording.log")
DESKTOP_LOG = "outputs/dashboard_logs/dashboard_desktop.log"
DEFAULT_REMOTE = "cal@desktop"
DEFAULT_REMOTE_WORKDIR = "~/lerobot-go-train/lerobot"
DEFAULT_REMOTE_UV = "/home/cal/.local/bin/uv"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, check=check)


def shell(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, shell=True, check=check)


def quote_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def dashboard_command(
    config: Path,
    *,
    follower_port: str | None = None,
    leader_port: str | None = None,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        str(GO_BOARD_DIR / "recording_dashboard.py"),
        "--config",
        str(config),
    ]
    if follower_port:
        parts.extend(["--so101-port", follower_port])
    if leader_port:
        parts.extend(["--leader-port", leader_port])
    return quote_command(parts)


def start_laptop(args: argparse.Namespace) -> None:
    command = dashboard_command(
        LAPTOP_CONFIG,
        follower_port=args.follower_port,
        leader_port=args.leader_port,
    )
    if args.foreground:
        raise SystemExit(shell(command).returncode)

    LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    stop_local_processes(args.session)
    screen_command = f"cd {shlex.quote(str(REPO_ROOT))} && {command} > {shlex.quote(str(LOCAL_LOG))} 2>&1"
    run(["screen", "-dmS", args.session, "/bin/zsh", "-lc", screen_command])
    print(f"Started laptop dashboard in screen session '{args.session}'.")
    print("URL: http://127.0.0.1:8766/")
    print(f"Log: {LOCAL_LOG}")


def stop_laptop(args: argparse.Namespace) -> None:
    stopped = stop_local_processes(args.session)
    if stopped:
        print(f"Stopped laptop dashboard screen session '{args.session}'.")
    else:
        print(f"No laptop dashboard screen session named '{args.session}' was running.")


def stop_local_processes(session: str) -> bool:
    result = shell(f"screen -S {shlex.quote(session)} -X quit >/dev/null 2>&1", check=False)
    shell("pkill -f '[r]ecording_dashboard.py' >/dev/null 2>&1 || true", check=False)
    return result.returncode == 0


def status_laptop(args: argparse.Namespace) -> None:
    shell("screen -ls || true", check=False)
    print_dashboard_state("http://127.0.0.1:8766/api/state")


def remote_prefix(args: argparse.Namespace) -> list[str]:
    return ["ssh", args.remote]


def remote_start_command(args: argparse.Namespace) -> str:
    remote_dashboard = quote_command(
        [
            args.remote_uv,
            "run",
            "python",
            str(GO_BOARD_DIR / "recording_dashboard.py"),
            "--config",
            str(DESKTOP_CONFIG),
            "--host",
            "0.0.0.0",
            "--port",
            "8766",
        ]
    )
    return (
        f"cd {shlex.quote(args.remote_workdir)} && "
        "mkdir -p outputs/dashboard_logs && "
        f"nohup {remote_dashboard} > {shlex.quote(DESKTOP_LOG)} 2>&1 < /dev/null & echo $!"
    )


def start_desktop(args: argparse.Namespace) -> None:
    remote_command = remote_start_command(args)
    run([*remote_prefix(args), remote_command])
    print("Started desktop dashboard.")
    print("URL: http://desktop:8766/")
    print(f"Remote log: {args.remote_workdir}/{DESKTOP_LOG}")


def stop_desktop(args: argparse.Namespace) -> None:
    command = "pkill -f '[r]ecording_dashboard.py' || true"
    run([*remote_prefix(args), command])
    print("Stopped desktop dashboard processes matching recording_dashboard.py.")


def status_desktop(args: argparse.Namespace) -> None:
    command = "pgrep -af '[r]ecording_dashboard.py' || true"
    run([*remote_prefix(args), command])
    print_dashboard_state("http://desktop:8766/api/state")


def sync_desktop_configs(args: argparse.Namespace) -> None:
    target = f"{args.remote}:{args.remote_workdir.rstrip('/')}/examples/go_board/"
    run(
        [
            "rsync",
            "-az",
            str(LAPTOP_CONFIG),
            str(DESKTOP_CONFIG),
            str(GO_BOARD_DIR / "manage_dashboard.py"),
            target,
        ]
    )
    print(f"Synced dashboard profiles to {target}")


def print_dashboard_state(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Dashboard API unavailable at {url}: {exc}")
        return
    cameras = ", ".join(
        f"{camera.get('name')} fresh={camera.get('fresh')} fps={camera.get('measured_fps')}"
        for camera in data.get("cameras", [])
    )
    print(f"connected={data.get('connected')} mode={data.get('mode')} teleop={data.get('teleop_enabled')}")
    print(f"note={data.get('note')}")
    print(f"cameras={cameras}")
    print(f"recording={data.get('recording', {}).get('message')}")


def add_remote_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-workdir", default=DEFAULT_REMOTE_WORKDIR)
    parser.add_argument("--remote-uv", default=DEFAULT_REMOTE_UV)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    laptop = subparsers.add_parser("laptop", help="Start the laptop dashboard for teleoperation recording.")
    laptop.add_argument("--session", default=LOCAL_SESSION)
    laptop.add_argument("--foreground", action="store_true")
    laptop.add_argument("--follower-port", help="Override the follower serial port for this run.")
    laptop.add_argument("--leader-port", help="Override the leader serial port for this run.")
    laptop.set_defaults(func=start_laptop)

    stop_local = subparsers.add_parser("stop-laptop", help="Stop the laptop dashboard screen session.")
    stop_local.add_argument("--session", default=LOCAL_SESSION)
    stop_local.set_defaults(func=stop_laptop)

    status_local = subparsers.add_parser("status-laptop", help="Show laptop dashboard status.")
    status_local.add_argument("--session", default=LOCAL_SESSION)
    status_local.set_defaults(func=status_laptop)

    desktop = subparsers.add_parser("desktop", help="Start the dashboard on the desktop over SSH.")
    add_remote_args(desktop)
    desktop.set_defaults(func=start_desktop)

    stop_remote = subparsers.add_parser("stop-desktop", help="Stop desktop dashboard processes over SSH.")
    add_remote_args(stop_remote)
    stop_remote.set_defaults(func=stop_desktop)

    status_remote = subparsers.add_parser("status-desktop", help="Show desktop dashboard status.")
    add_remote_args(status_remote)
    status_remote.set_defaults(func=status_desktop)

    sync_remote = subparsers.add_parser("sync-desktop-configs", help="Copy dashboard profile files to the desktop.")
    add_remote_args(sync_remote)
    sync_remote.set_defaults(func=sync_desktop_configs)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
