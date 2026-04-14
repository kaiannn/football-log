"""Web UI 入口。

普通模式:  python3 run_web.py
热更新:    python3 run_web.py --reload  (修改代码后自动重载)
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Football Log Web UI")
    parser.add_argument("--reload", action="store_true", help="启用热更新")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.reload:
        project_root = os.path.dirname(os.path.abspath(__file__))
        env = os.environ.copy()
        env["GRADIO_SERVER_PORT"] = str(args.port)
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [
            sys.executable, "-m", "gradio",
            "football_log/app/web.py",
            "--demo-name", "demo",
        ]
        subprocess.run(cmd, env=env, cwd=project_root)
    else:
        from football_log.app.web import demo
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.port,
        )


if __name__ == "__main__":
    main()
