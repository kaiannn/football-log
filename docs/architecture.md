# football-log 架构流程图

## 主流水线

```mermaid
graph TD
    subgraph 入口
        A[run.py] -->|调用| B[app/cli.py main]
        C[run_web.py] -->|import| D[app/web.py demo]
    end

    B -->|构建| E[VideoTrackerPipeline]
    D -->|构建| E

    subgraph runner.py — VideoTrackerPipeline
        E --> F{组件初始化}
        F -->|默认| G[YoloByteTrackTracker]
        F -->|--tracker deepsort| H[DeepSortTracker]
        F -->|--team-class-model| I[6-class YOLO]
        F -->|默认| J[TeamClassifier HSV]
        F -->|--team-classifier keypoint| K[KeypointTeamClassifier]
        F -->|默认| L[TrackingDataWriter]
        F -->|--homography| M[HomographyProjector]
        F -->|--camera-calib| N[PinholeGroundProjector]
        F -->|--auto-calibration-keyframes| O[AutoCalibrationProjector]
        F -->|--bev-smoothing| P[TrackFilter + jump_likelihood]
        F -->|--pitch-field-detect| Q[PitchFieldEstimator]
        F -->|--pitch-keypoint-model| R[PitchKeypointDetector]
        F -->|--ball-model| S[BallDetector]
        F -->|--save-radar| T[RadarRenderer]
    end

    subgraph _iter_frames 每帧循环
        U[读取帧] --> V{应检测?}
        V -->|是| W[detector.detect]
        W --> X[team_classifier 分队]
        V -->|否| Y[复用上次检测]
        X --> Z{有 projector?}
        Y --> Z
        Z -->|是| AA[projector.project → world_x/y]
        Z -->|否| AB{有 _kp_H?}
        AB -->|是| AC[perspectiveTransform → world_x/y]
        AA --> AD{pitch_field_detect?}
        AC --> AD
        AD -->|是| AE[PitchFieldEstimator.estimate]
        AE --> AF[filter_objects_in_grass_mask]
        AD --> AG{有 _pitch_kp_detector?}
        AG -->|是| AH[PitchKeypointDetector.detect → _kp_H EMA]
        AI{有 _ball_detector?} -->|是| AJ[BallDetector.detect → 覆盖球检测]
        AK{有 _track_filter?} -->|是| AL[jump_likelihood_from_height_change + Kalman 平滑]
        AM[TrackingDataWriter.write_frame]
        AN{save_debug_overlay?} -->|是| AO[draw_pitch_observation<br/>草地掩膜 + 场线 + 四边形]
        AN --> AP[draw_tracking_overlay + draw_frame_hud]
        AQ{save_radar?} -->|是| AR[RadarRenderer.render]
    end
```

## 子包依赖关系

```mermaid
graph LR
    subgraph 入口层
        run_py[run.py]
        run_web[run_web.py]
        scripts[scripts/]
    end

    subgraph 应用层
        cli[app/cli.py]
        web[app/web.py]
        runner[app/runner.py]
        calib_ref[app/calibrate_reference.py]
    end

    subgraph 接口层
        proto[protocols.py<br/>Detection + 7 Protocol]
    end

    subgraph 视觉层
        tracker[vision/tracker.py<br/>YoloByteTrackTracker]
        deepsort[vision/deepsort_tracker.py<br/>DeepSortTracker]
        tc[vision/team_classifier.py<br/>TeamClassifier]
        tc_kp[vision/team_classifier_keypoint.py<br/>KeypointTeamClassifier]
        pose[vision/pose.py<br/>PoseEstimator]
        pkd[vision/pitch_keypoint_detector.py<br/>PitchKeypointDetector]
        ball[vision/ball_detector.py<br/>BallDetector]
        reg[vision/tracker_registry.py<br/>resolve_tracker]
        reid[vision/reid.py<br/>ReIDExtractor Protocol]
    end

    subgraph 世界坐标层
        homo[world/homography.py<br/>Homography + Projector]
        pinhole[world/pinhole_ground.py<br/>PinholeGroundProjector]
        auto_cal[world/auto_calibration.py<br/>AutoCalibrationProjector]
        track_f[world/track_filter.py<br/>TrackFilter + jump]
        pitch_m[world/pitch_model.py<br/>PitchSpec]
        heur_ref[world/heuristic_reference_fit.py]
        validate[world/validate_projection.py]
    end

    subgraph 场地层
        fe[pitch/field_estimator.py<br/>PitchFieldEstimator]
        integ[pitch/integration.py<br/>filter_objects_in_grass_mask]
        obs[pitch/observation.py<br/>PitchObservation]
        cfg[pitch/config.py<br/>PitchFieldConfig]
    end

    subgraph 导出层
        exp[io/export.py<br/>TrackingDataWriter]
    end

    subgraph UI层
        overlay[ui/overlay.py<br/>draw_tracking_overlay<br/>draw_pitch_observation<br/>draw_frame_hud]
        radar[ui/radar.py<br/>RadarRenderer]
    end

    subgraph 评估层
        eval_det[eval/eval_detection.py]
        eval_track[eval/eval_tracking.py]
        eval_world[eval/eval_world.py]
        eval_report[eval/report.py]
        eval_run[eval/run_all.py]
    end

    subgraph 数据准备
        gsr[data/gsr_convert.py]
        yolo_c[data/yolo_convert.py]
    end

    subgraph 实验层
        tvcalib[experiments/tvcalib_infer.py]
        roboflow[experiments/roboflow_sports_radar.py]
    end

    run_py --> cli
    run_web --> web
    cli --> runner
    web --> runner
    runner --> proto
    runner --> tracker
    runner --> deepsort
    runner --> tc
    runner --> tc_kp
    runner --> pkd
    runner --> ball
    runner --> homo
    runner --> pinhole
    runner --> auto_cal
    runner --> track_f
    runner --> pitch_m
    runner --> fe
    runner --> integ
    runner --> exp
    runner --> overlay
    runner --> radar
    tracker --> reg
    tracker --> tc
    deepsort --> tc
    tc_kp --> pose
    auto_cal --> homo
    pinhole --> homo
    calib_ref --> heur_ref
    calib_ref --> homo
    scripts --> gsr
    scripts --> yolo_c
    eval_run --> eval_det
    eval_run --> eval_track
    eval_run --> eval_world
    eval_world --> homo
    eval_world --> pinhole
```

## 独立工具（不被主流水线调用）

| 模块 | 功能 | 用途 |
|------|------|------|
| `vision/reid.py` | `ReIDExtractor` Protocol + `cosine_similarity` | #12 Re-ID 扩展预留接口 |
| `world/heuristic_reference_fit.py` | 启发式标定物拟合 | `calibrate_reference.py` 独立标定工具 |
| `world/validate_projection.py` | 投影质量验证（重投影/物理约束/已知点） | 独立 CLI 验证工具 |
| `app/calibrate_reference.py` | 交互式/文件式标定物拟合 | 独立 CLI 标定工具 |
| `eval/` | 检测/跟踪/世界坐标评估 + 报告生成 | 独立评估脚本（subprocess 调用） |
| `data/` | SoccerNet → YOLO 格式转换 | `scripts/prepare_yolo_dataset.py` |
| `experiments/` | TVCalib / Roboflow Sports 环境搭建 | 外部工具集成脚本 |
