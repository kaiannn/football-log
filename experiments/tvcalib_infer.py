from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TVCALIB_REPO_DIR = PROJECT_ROOT / ".cache" / "tvcalib"
DEFAULT_REPO_URL = "https://github.com/mm4spa/tvcalib.git"
DEFAULT_WEIGHTS_URL = "https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt"


def _run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path, repo_url: str) -> Path:
    if (repo_dir / ".git").is_dir():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_checked(["git", "clone", repo_url, str(repo_dir)])
    return repo_dir


def ensure_conda_env(repo_dir: Path, env_name: str) -> None:
    env_file = repo_dir / "environment.yml"
    if not env_file.is_file():
        raise SystemExit(f"未找到 TVCalib environment.yml: {env_file}")
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
    if env_name not in result.stdout:
        _run_checked(["conda", "env", "create", "-n", env_name, "-f", str(env_file)], cwd=repo_dir)


def ensure_segment_weights(repo_dir: Path, weights_url: str, weights_path: Path | None) -> Path:
    target = weights_path or (repo_dir / "data" / "segment_localization" / "train_59.pt")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(weights_url, target)
    return target


def launch_notebook(repo_dir: Path, env_name: str) -> None:
    notebook = repo_dir / "inference.ipynb"
    if not notebook.is_file():
        raise SystemExit(f"未找到 TVCalib inference notebook: {notebook}")
    cmd = f"conda run -n {env_name} jupyter notebook {notebook}"
    _run_checked(["bash", "-lc", cmd], cwd=repo_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="准备 TVCalib 官方 inference 环境并打开 notebook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-dir", default=str(TVCALIB_REPO_DIR), help="本地缓存的 TVCalib 仓库路径")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="TVCalib 仓库地址")
    parser.add_argument("--env-name", default="tvcalib", help="conda 环境名")
    parser.add_argument("--weights-url", default=DEFAULT_WEIGHTS_URL, help="官方 segmentation 权重下载地址")
    parser.add_argument("--weights-path", default="", help="已下载权重路径；为空则下载到仓库 data/segment_localization/train_59.pt")
    parser.add_argument("--skip-install", action="store_true", help="跳过 clone / conda env create / 权重下载")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_dir = Path(os.path.abspath(args.repo_dir))
    weights_path = Path(os.path.abspath(args.weights_path)) if args.weights_path else None

    print(f"TVCalib 仓库: {repo_dir}")
    print(f"conda 环境: {args.env_name}")

    if not args.skip_install:
        ensure_repo(repo_dir, args.repo_url)
        ensure_conda_env(repo_dir, args.env_name)
        final_weights = ensure_segment_weights(repo_dir, args.weights_url, weights_path)
        print(f"分割权重: {final_weights}")

    launch_notebook(repo_dir, args.env_name)


if __name__ == "__main__":
    main()
