from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPORTS_REPO_DIR = PROJECT_ROOT / ".cache" / "sports"
DEFAULT_REPO_URL = "https://github.com/roboflow/sports.git"


def _run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path, repo_url: str) -> Path:
    if (repo_dir / ".git").is_dir():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(["git", "clone", repo_url, str(repo_dir)])
    return repo_dir


def ensure_example_requirements(repo_dir: Path) -> None:
    req = repo_dir / "examples" / "soccer" / "requirements.txt"
    if not req.is_file():
        raise SystemExit(f"未找到 Roboflow Sports 示例依赖文件: {req}")
    _run_checked([sys.executable, "-m", "pip", "install", "-r", str(req)], cwd=repo_dir)


def _setup_script(repo_dir: Path) -> Path:
    script = repo_dir / "examples" / "soccer" / "setup.sh"
    if not script.is_file():
        raise SystemExit(f"未找到 Roboflow Sports setup 脚本: {script}")
    return script


def ensure_setup(repo_dir: Path) -> None:
    script = _setup_script(repo_dir)
    _run_checked(["bash", str(script)], cwd=script.parent)


def run_radar(repo_dir: Path, source: str, target: str, device: str) -> None:
    example_dir = repo_dir / "examples" / "soccer"
    main_py = example_dir / "main.py"
    if not main_py.is_file():
        raise SystemExit(f"未找到 Roboflow Sports 示例入口: {main_py}")
    _run_checked(
        [
            sys.executable,
            str(main_py),
            "--source_video_path",
            source,
            "--target_video_path",
            target,
            "--device",
            device,
            "--mode",
            "RADAR",
        ],
        cwd=example_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 Roboflow Sports 官方 soccer RADAR 示例",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument("--target", default=None, help="输出视频路径，默认 <source>_roboflow_radar.mp4")
    parser.add_argument("--device", default="cpu", help="推理设备，例如 cpu / mps / cuda")
    parser.add_argument("--repo-dir", default=str(SPORTS_REPO_DIR), help="本地缓存的 sports 仓库路径")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="sports 仓库地址")
    parser.add_argument("--skip-install", action="store_true", help="跳过 clone / pip install / setup")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        raise SystemExit(f"输入视频不存在: {source}")

    if args.target:
        target = os.path.abspath(args.target)
    else:
        stem, ext = os.path.splitext(source)
        target = f"{stem}_roboflow_radar{ext or '.mp4'}"

    repo_dir = Path(os.path.abspath(args.repo_dir))

    print(f"输入: {source}")
    print(f"输出: {target}")
    print(f"设备: {args.device}")
    print(f"sports 仓库: {repo_dir}")

    if not args.skip_install:
        ensure_repo(repo_dir, args.repo_url)
        ensure_example_requirements(repo_dir)
        ensure_setup(repo_dir)

    run_radar(repo_dir, source, target, args.device)
    print(f"完成: {target}")


if __name__ == "__main__":
    main()
