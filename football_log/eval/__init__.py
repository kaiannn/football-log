"""离线评估脚本：detection / tracking / world projection 三类指标。

每个子模块都可作为 `python -m football_log.eval.<name>` 直接运行，
输出 JSON 格式的 metrics 到 `--out` 指定路径，便于跨实验对比。
"""
