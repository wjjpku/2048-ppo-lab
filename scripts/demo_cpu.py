"""One-command local CPU demonstration; creates only a project-local venv/output."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import ssl
import venv

def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Python3.11 or newer is required")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=root / ".venv",
                        help="optional existing/test virtual environment")
    target = parser.parse_args().venv.resolve()
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(target)
    install_env = os.environ.copy()
    # Some macOS Python installers leave their bundled CA path unconfigured.
    # Use the system CA store for these subprocesses; keep TLS verification on.
    if (sys.platform == "darwin" and not ssl.get_default_verify_paths().cafile
            and Path("/etc/ssl/cert.pem").is_file()):
        install_env.setdefault("PIP_CERT", "/etc/ssl/cert.pem")
    command = [str(python), "-m", "pip", "install", "torch==2.11.0"]
    if sys.platform.startswith("linux") or os.name == "nt":
        command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
    subprocess.run(command, check=True, cwd=root, env=install_env)
    subprocess.run([str(python), "-m", "pip", "install", "-e", "."], check=True, cwd=root, env=install_env)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "demo-output" / stamp
    subprocess.run([str(python), "-m", "game2048", "demo", "--device", "cpu", "--output", str(output)], check=True, cwd=root)
    print(f"Open the complete replay: {output / 'replay.html'}")

if __name__ == "__main__":
    main()
