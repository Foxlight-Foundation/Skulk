"""Build a local macOS app bundle for Local Network attribution probes."""

from __future__ import annotations

import argparse
import json
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast, final

from pydantic import Field

from skulk.utils.pydantic_ext import FrozenModel

DEFAULT_BUNDLE_IDENTIFIER = "foundation.foxlight.skulk.local-network-probe"
DEFAULT_DISPLAY_NAME = "Skulk Local Network Probe"
DEFAULT_EXECUTABLE_NAME = "SkulkLocalNetworkProbe"
DEFAULT_OUTPUT_APP = Path("/tmp/SkulkLocalNetworkProbe.app")
DEFAULT_LOG_DIR = Path("~/Library/Logs/SkulkLocalNetworkProbe")
LOCAL_NETWORK_USAGE_DESCRIPTION = (
    "Skulk uses the local network to discover and connect to nearby Skulk "
    "nodes, including Thunderbolt-connected peers."
)
LauncherKind = Literal["native", "script"]


@final
class MacOSLocalNetworkProbeAppBuild(FrozenModel):
    """Result of building a throwaway macOS Local Network probe app.

    Attributes:
        app_path: Absolute path to the generated ``.app`` bundle.
        executable_path: Absolute path to the bundle executable.
        info_plist_path: Absolute path to the generated ``Info.plist``.
        log_dir: Directory where the app writes the probe output.
        launcher_kind: Launcher implementation used inside ``Contents/MacOS``.
        ad_hoc_signed: Whether the app was successfully ad-hoc code signed.
        codesign_message: Warning or error text from codesign, when present.
    """

    app_path: str = Field(description="Absolute path to the generated .app bundle.")
    executable_path: str = Field(
        description="Absolute path to the bundle executable.",
    )
    info_plist_path: str = Field(
        description="Absolute path to the generated Info.plist.",
    )
    log_dir: str = Field(description="Directory where the app writes probe output.")
    launcher_kind: LauncherKind = Field(
        description="Launcher implementation used inside Contents/MacOS.",
    )
    ad_hoc_signed: bool = Field(
        description="Whether the app was successfully ad-hoc code signed.",
    )
    codesign_message: str | None = Field(
        default=None,
        description="Warning or error text from codesign, when present.",
    )


def _info_plist(
    *,
    bundle_identifier: str,
    display_name: str,
    executable_name: str,
) -> dict[str, object]:
    return {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": display_name,
        "CFBundleExecutable": executable_name,
        "CFBundleIdentifier": bundle_identifier,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": display_name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "15.0",
        "NSLocalNetworkUsageDescription": LOCAL_NETWORK_USAGE_DESCRIPTION,
    }


def _launcher_script(*, repo_root: Path, log_dir: Path) -> str:
    quoted_repo_root = shlex.quote(str(repo_root))
    quoted_log_dir = shlex.quote(str(log_dir.expanduser()))
    return f"""#!/usr/bin/env bash
set -u

REPO_ROOT={quoted_repo_root}
DEFAULT_LOG_DIR={quoted_log_dir}
LOG_DIR="${{SKULK_LOCAL_NETWORK_PROBE_LOG_DIR:-$DEFAULT_LOG_DIR}}"
JSON_OUT="${{LOG_DIR}}/latest.json"
STDERR_OUT="${{LOG_DIR}}/latest.stderr.log"
STATUS_OUT="${{LOG_DIR}}/latest.status"

mkdir -p "$LOG_DIR" || exit 73
cd "$REPO_ROOT" || exit 70

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
elif [[ -x "/opt/homebrew/bin/uv" ]]; then
  UV_BIN="/opt/homebrew/bin/uv"
else
  printf 'uv not found in PATH, ~/.local/bin, or /opt/homebrew/bin\\n' >"$STDERR_OUT"
  printf '%s\\n' '127' >"$STATUS_OUT"
  exit 127
fi

"$UV_BIN" run skulk-macos-local-network-probe --json >"$JSON_OUT" 2>"$STDERR_OUT"
STATUS=$?
printf '%s\\n' "$STATUS" >"$STATUS_OUT"
exit "$STATUS"
"""


def _native_launcher_source(*, repo_root: Path, log_dir: Path) -> str:
    repo_root_literal = json.dumps(str(repo_root))
    log_dir_literal = json.dumps(str(log_dir.expanduser()))
    probe_command_literal = json.dumps(
        str(repo_root / ".venv" / "bin" / "skulk-macos-local-network-probe")
    )
    return f"""#import <AppKit/AppKit.h>

#include <arpa/inet.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static const char *REPO_ROOT = {repo_root_literal};
static const char *DEFAULT_LOG_DIR = {log_dir_literal};
static const char *PROBE_COMMAND = {probe_command_literal};
static const int PROBE_PORT = 9;

static int join_path(char *destination, size_t destination_size, const char *directory, const char *filename) {{
    int written = snprintf(destination, destination_size, "%s/%s", directory, filename);
    return written < 0 || (size_t)written >= destination_size ? -1 : 0;
}}

static int mkdir_p(const char *path) {{
    char buffer[PATH_MAX];
    size_t length = strnlen(path, sizeof(buffer));
    if (length == 0 || length >= sizeof(buffer)) {{
        return -1;
    }}

    memcpy(buffer, path, length + 1);
    if (buffer[length - 1] == '/') {{
        buffer[length - 1] = '\\0';
    }}

    for (char *cursor = buffer + 1; *cursor != '\\0'; cursor++) {{
        if (*cursor != '/') {{
            continue;
        }}
        *cursor = '\\0';
        if (mkdir(buffer, 0755) != 0 && errno != EEXIST) {{
            return -1;
        }}
        *cursor = '/';
    }}

    return mkdir(buffer, 0755) == 0 || errno == EEXIST ? 0 : -1;
}}

static int default_gateway_ipv4(char *gateway, size_t gateway_size) {{
    FILE *route_output = popen("/sbin/route -n get default 2>/dev/null", "r");
    if (route_output == NULL) {{
        return -1;
    }}

    char line[256];
    int found = -1;
    while (fgets(line, sizeof(line), route_output) != NULL) {{
        char parsed_gateway[128];
        if (sscanf(line, " gateway: %127s", parsed_gateway) == 1) {{
            int written = snprintf(gateway, gateway_size, "%s", parsed_gateway);
            found = written >= 0 && (size_t)written < gateway_size ? 0 : -1;
            break;
        }}
    }}
    pclose(route_output);
    return found;
}}

static const char *probe_local_network(const char *gateway, int *probe_errno) {{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {{
        *probe_errno = errno;
        return "unknown";
    }}

    struct timeval timeout;
    timeout.tv_sec = 2;
    timeout.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_port = htons(PROBE_PORT);
    if (inet_pton(AF_INET, gateway, &address.sin_addr) != 1) {{
        close(fd);
        *probe_errno = EINVAL;
        return "unknown";
    }}

    int result = connect(fd, (struct sockaddr *)&address, sizeof(address));
    if (result == 0) {{
        close(fd);
        *probe_errno = 0;
        return "ok";
    }}

    int connect_errno = errno;
    close(fd);
    *probe_errno = connect_errno;
    return connect_errno == EHOSTUNREACH ? "blocked" : "ok";
}}

static void write_launcher_probe_file(const char *path) {{
    char gateway[128];
    const char *status = "unknown";
    int probe_errno = 0;
    if (default_gateway_ipv4(gateway, sizeof(gateway)) == 0) {{
        status = probe_local_network(gateway, &probe_errno);
    }} else {{
        snprintf(gateway, sizeof(gateway), "%s", "");
    }}

    FILE *file = fopen(path, "w");
    if (file != NULL) {{
        fprintf(
            file,
            "{{\\n"
            "  \\"local_network_status\\": \\"%s\\",\\n"
            "  \\"gateway\\": \\"%s\\",\\n"
            "  \\"errno\\": %d,\\n"
            "  \\"pid\\": %d,\\n"
            "  \\"process\\": \\"SkulkLocalNetworkProbe\\"\\n"
            "}}\\n",
            status,
            gateway,
            probe_errno,
            getpid()
        );
        fclose(file);
    }}
}}

static void write_status_file(const char *status_path, int status) {{
    FILE *file = fopen(status_path, "w");
    if (file != NULL) {{
        fprintf(file, "%d\\n", status);
        fclose(file);
    }}
}}

static int run_probe(void) {{
    const char *log_dir = getenv("SKULK_LOCAL_NETWORK_PROBE_LOG_DIR");
    if (log_dir == NULL || log_dir[0] == '\\0') {{
        log_dir = DEFAULT_LOG_DIR;
    }}

    if (mkdir_p(log_dir) != 0) {{
        return 73;
    }}

    char json_path[PATH_MAX];
    char launcher_probe_path[PATH_MAX];
    char stderr_path[PATH_MAX];
    char status_path[PATH_MAX];
    if (
        join_path(json_path, sizeof(json_path), log_dir, "latest.json") != 0 ||
        join_path(launcher_probe_path, sizeof(launcher_probe_path), log_dir, "launcher-preflight.json") != 0 ||
        join_path(stderr_path, sizeof(stderr_path), log_dir, "latest.stderr.log") != 0 ||
        join_path(status_path, sizeof(status_path), log_dir, "latest.status") != 0
    ) {{
        return 75;
    }}

    write_launcher_probe_file(launcher_probe_path);

    int json_fd = open(json_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    int stderr_fd = open(stderr_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (json_fd < 0 || stderr_fd < 0) {{
        if (json_fd >= 0) {{
            close(json_fd);
        }}
        if (stderr_fd >= 0) {{
            close(stderr_fd);
        }}
        return 74;
    }}

    if (access(PROBE_COMMAND, X_OK) != 0) {{
        dprintf(stderr_fd, "probe command is not executable: %s: %s\\n", PROBE_COMMAND, strerror(errno));
        close(json_fd);
        close(stderr_fd);
        write_status_file(status_path, 127);
        return 127;
    }}

    pid_t pid = fork();
    if (pid < 0) {{
        dprintf(stderr_fd, "fork failed: %s\\n", strerror(errno));
        close(json_fd);
        close(stderr_fd);
        write_status_file(status_path, 71);
        return 71;
    }}

    if (pid == 0) {{
        if (chdir(REPO_ROOT) != 0) {{
            dprintf(stderr_fd, "chdir failed: %s: %s\\n", REPO_ROOT, strerror(errno));
            _exit(70);
        }}
        if (dup2(json_fd, STDOUT_FILENO) < 0 || dup2(stderr_fd, STDERR_FILENO) < 0) {{
            _exit(74);
        }}
        close(json_fd);
        close(stderr_fd);

        char *const argv[] = {{
            (char *)PROBE_COMMAND,
            "--json",
            NULL,
        }};
        execv(PROBE_COMMAND, argv);
        fprintf(stderr, "execv failed: %s: %s\\n", PROBE_COMMAND, strerror(errno));
        _exit(127);
    }}

    close(json_fd);
    close(stderr_fd);

    int wait_status = 0;
    if (waitpid(pid, &wait_status, 0) < 0) {{
        write_status_file(status_path, 72);
        return 72;
    }}

    int exit_status = 1;
    if (WIFEXITED(wait_status)) {{
        exit_status = WEXITSTATUS(wait_status);
    }} else if (WIFSIGNALED(wait_status)) {{
        exit_status = 128 + WTERMSIG(wait_status);
    }}
    write_status_file(status_path, exit_status);
    return exit_status;
}}

int main(int argc, char *argv[]) {{
    (void)argc;
    (void)argv;
    @autoreleasepool {{
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp activateIgnoringOtherApps:NO];
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{{
            int status = run_probe();
            exit(status);
        }});
        [NSApp run];
    }}
    return 0;
}}
"""


def _compile_native_launcher(
    *,
    source_path: Path,
    executable_path: Path,
) -> tuple[bool, str | None]:
    clang = shutil.which("clang")
    if clang is None:
        return False, "clang not found"

    completed = subprocess.run(
        [
            clang,
            "-Wall",
            "-Wextra",
            "-O2",
            "-fobjc-arc",
            "-framework",
            "AppKit",
            str(source_path),
            "-o",
            str(executable_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True, None

    message = completed.stderr.strip() or completed.stdout.strip()
    if not message:
        message = f"clang exited with status {completed.returncode}"
    return False, message


def _ad_hoc_sign(app_path: Path) -> tuple[bool, str | None]:
    codesign = shutil.which("codesign")
    if codesign is None:
        return False, "codesign not found"

    completed = subprocess.run(
        [codesign, "--force", "--deep", "--sign", "-", str(app_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True, None

    message = completed.stderr.strip() or completed.stdout.strip()
    if not message:
        message = f"codesign exited with status {completed.returncode}"
    return False, message


def build_macos_local_network_probe_app(
    *,
    output_app: Path,
    repo_root: Path,
    log_dir: Path = DEFAULT_LOG_DIR,
    bundle_identifier: str = DEFAULT_BUNDLE_IDENTIFIER,
    display_name: str = DEFAULT_DISPLAY_NAME,
    executable_name: str = DEFAULT_EXECUTABLE_NAME,
    launcher_kind: LauncherKind = "native",
    ad_hoc_sign: bool = True,
    replace_existing: bool = False,
) -> MacOSLocalNetworkProbeAppBuild:
    """Build a disposable macOS app bundle that launches the probe command.

    Args:
        output_app: Destination ``.app`` bundle path.
        repo_root: Skulk repository root that contains the ``uv`` environment.
        log_dir: Directory where the app should write ``latest.json``.
        bundle_identifier: Stable bundle identifier for macOS privacy identity.
        display_name: Human-facing app name used by Finder and privacy prompts.
        executable_name: Name of the executable inside ``Contents/MacOS``.
        launcher_kind: Use a native Mach-O launcher or a shell-script control.
        ad_hoc_sign: Attempt to ad-hoc sign the generated app bundle.
        replace_existing: Remove an existing app bundle at ``output_app`` first.

    Returns:
        Metadata for the generated app bundle.

    Raises:
        FileExistsError: If ``output_app`` exists and replacement is disabled.
        ValueError: If ``output_app`` is not a ``.app`` bundle path.
    """

    resolved_output_app = output_app.expanduser().resolve()
    resolved_repo_root = repo_root.expanduser().resolve()
    expanded_log_dir = log_dir.expanduser()
    if resolved_output_app.suffix != ".app":
        msg = f"output path must end with .app: {resolved_output_app}"
        raise ValueError(msg)

    if resolved_output_app.exists():
        if not replace_existing:
            msg = f"{resolved_output_app} already exists; pass --replace to rebuild it"
            raise FileExistsError(msg)
        shutil.rmtree(resolved_output_app)

    contents_dir = resolved_output_app / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    info_plist_path = contents_dir / "Info.plist"
    executable_path = macos_dir / executable_name

    macos_dir.mkdir(parents=True)
    resources_dir.mkdir()
    with info_plist_path.open("wb") as file:
        plistlib.dump(
            _info_plist(
                bundle_identifier=bundle_identifier,
                display_name=display_name,
                executable_name=executable_name,
            ),
            file,
        )

    if launcher_kind == "native":
        launcher_source_path = resources_dir / "launcher.m"
        launcher_source_path.write_text(
            _native_launcher_source(
                repo_root=resolved_repo_root,
                log_dir=expanded_log_dir,
            ),
            encoding="utf-8",
        )
        compiled, compile_message = _compile_native_launcher(
            source_path=launcher_source_path,
            executable_path=executable_path,
        )
        if not compiled:
            msg = f"failed to compile native launcher: {compile_message}"
            raise RuntimeError(msg)
    else:
        executable_path.write_text(
            _launcher_script(repo_root=resolved_repo_root, log_dir=expanded_log_dir),
            encoding="utf-8",
        )
    executable_mode = executable_path.stat().st_mode
    executable_path.chmod(
        executable_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    ad_hoc_signed = False
    codesign_message: str | None = None
    if ad_hoc_sign:
        ad_hoc_signed, codesign_message = _ad_hoc_sign(resolved_output_app)

    return MacOSLocalNetworkProbeAppBuild(
        app_path=str(resolved_output_app),
        executable_path=str(executable_path),
        info_plist_path=str(info_plist_path),
        log_dir=str(expanded_log_dir),
        launcher_kind=launcher_kind,
        ad_hoc_signed=ad_hoc_signed,
        codesign_message=codesign_message,
    )


def format_macos_local_network_probe_app_build(
    result: MacOSLocalNetworkProbeAppBuild,
) -> str:
    """Format app build output with the commands needed for the experiment.

    Args:
        result: Build result to format.

    Returns:
        Multi-line human-readable instructions.
    """

    quoted_app_path = shlex.quote(result.app_path)
    quoted_json_path = shlex.quote(str(Path(result.log_dir).expanduser() / "latest.json"))
    lines = [
        f"Built {result.app_path}",
        f"Executable: {result.executable_path}",
        f"Info.plist: {result.info_plist_path}",
        f"Probe output: {Path(result.log_dir).expanduser() / 'latest.json'}",
        f"Launcher preflight: {Path(result.log_dir).expanduser() / 'launcher-preflight.json'}",
        f"Launcher: {result.launcher_kind}",
        f"Ad-hoc signed: {'yes' if result.ad_hoc_signed else 'no'}",
        "",
        "Run:",
        f"  open -W {quoted_app_path}",
        "",
        "Inspect:",
        f"  cat {quoted_json_path}",
    ]
    if result.codesign_message is not None:
        lines.extend(["", f"codesign: {result.codesign_message}"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the macOS app bundle from the command line.

    Args:
        argv: Optional argument vector, excluding program name.

    Returns:
        Process exit status.
    """

    parser = argparse.ArgumentParser(
        prog="skulk-build-macos-local-network-probe-app",
        description="Build a disposable Skulk.app-style Local Network probe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_APP,
        help=f"Destination .app bundle path. Defaults to {DEFAULT_OUTPUT_APP}.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Skulk repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Probe output directory. Defaults to {DEFAULT_LOG_DIR}.",
    )
    parser.add_argument(
        "--bundle-identifier",
        default=DEFAULT_BUNDLE_IDENTIFIER,
        help=f"Bundle identifier. Defaults to {DEFAULT_BUNDLE_IDENTIFIER}.",
    )
    parser.add_argument(
        "--display-name",
        default=DEFAULT_DISPLAY_NAME,
        help=f"Display name. Defaults to {DEFAULT_DISPLAY_NAME}.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing app bundle at --output.",
    )
    parser.add_argument(
        "--launcher",
        choices=("native", "script"),
        default="native",
        help="Launcher implementation. Defaults to native.",
    )
    parser.add_argument(
        "--no-ad-hoc-sign",
        action="store_true",
        help="Skip ad-hoc codesigning of the generated app bundle.",
    )
    parsed_values = cast("dict[str, object]", vars(parser.parse_args(argv)))

    output = parsed_values.get("output", DEFAULT_OUTPUT_APP)
    repo_root = parsed_values.get("repo_root", Path.cwd())
    log_dir = parsed_values.get("log_dir", DEFAULT_LOG_DIR)
    if not isinstance(output, Path):
        output = DEFAULT_OUTPUT_APP
    if not isinstance(repo_root, Path):
        repo_root = Path.cwd()
    if not isinstance(log_dir, Path):
        log_dir = DEFAULT_LOG_DIR
    launcher_raw = parsed_values.get("launcher", "native")
    launcher_kind: LauncherKind = "script" if launcher_raw == "script" else "native"
    bundle_identifier_raw = parsed_values.get(
        "bundle_identifier",
        DEFAULT_BUNDLE_IDENTIFIER,
    )
    display_name_raw = parsed_values.get("display_name", DEFAULT_DISPLAY_NAME)
    bundle_identifier = (
        bundle_identifier_raw
        if isinstance(bundle_identifier_raw, str)
        else DEFAULT_BUNDLE_IDENTIFIER
    )
    display_name = (
        display_name_raw if isinstance(display_name_raw, str) else DEFAULT_DISPLAY_NAME
    )

    result = build_macos_local_network_probe_app(
        output_app=output,
        repo_root=repo_root,
        log_dir=log_dir,
        bundle_identifier=bundle_identifier,
        display_name=display_name,
        launcher_kind=launcher_kind,
        ad_hoc_sign=parsed_values.get("no_ad_hoc_sign") is not True,
        replace_existing=parsed_values.get("replace") is True,
    )
    print(format_macos_local_network_probe_app_build(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
