# Vehicle Release Acceptance

本文档定义 VCAS 学习版在目标 Windows 主机上的发布门禁。门禁策略由 `config/vehicle_release_acceptance.v1.json` 冻结，证据格式由 `api/schemas/vehicle_release_evidence.v1.schema.json` 冻结。

## 当前状态

M5 验收框架已经实现，但真实模型仍在云端训练。当前不得生成 `status=passed` 的 Release Evidence，也不得创建正式发布包。可以生成 `pending` 记录，用于冻结代码、配置哈希和未完成门禁。

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\new_vehicle_release_evidence.ps1 `
  -ReleaseId vehicle_v1_0_0_candidate
```

该命令会：

- 记录 Git Commit 和工作区是否干净。
- 记录 Server、Worker、Camera、车辆配置、标签、Registry 和数据库迁移的 SHA256。
- 执行模型交付审计。
- 生成 `reports/m5/<release_id>/release_evidence.json`。
- 以 `--allow-pending` 校验证据结构，但不会把阻塞项改写为通过。

## 发布门禁

只有以下全部满足，Evidence 才能设置为 `status=passed`：

1. 数据版本、标签、Model Registry、车辆配置、数据库 Schema 和代码 Commit 已冻结。
2. Detection/Attribute ONNX、Engine、SHA256、来源、模型卡和指标完整。
3. 原始模型、ONNX 与 TensorRT 在同一封闭测试集上的指标绝对下降不超过 0.005。
4. 两路 1920×1080 摄像头至少运行 30 分钟，每路 Detection 平均不低于 8 FPS，队列不持续增长。
5. 轨迹确认到属性稳定 P95 不高于 1500 ms，事件就绪到持久化 P95 不高于 2000 ms。
6. 连续运行至少 8 小时：零崩溃、无显存持续增长、无无界重连、无旧 Run 结果串入。
7. 每个 `(run_id, track_id)` 至多一个最终事件。
8. RTSP 断流、PostgreSQL 暂不可用、Redis 重启、回调 5xx 和 Worker 重启演练全部通过。
9. 数据库备份恢复、旧二进制、旧配置、旧 Engine 保留以及完整回滚演练全部通过。

严格门禁命令：

```powershell
python .\tools\validate_vehicle_release_evidence.py `
  .\reports\m5\vehicle_v1_0_0\release_evidence.json
```

`pending`、缺字段、低于边界值、不安全证据路径、非 SHA256、脏工作区或任一故障演练未通过都会失败关闭。

## 推荐执行顺序

### 1. 模型交付

按 `docs/models/MODEL_DELIVERY_STATUS.md` 完成真实模型回补：

```powershell
python .\tools\audit_model_delivery.py
```

### 2. 构建与集成测试

在一次性 PostgreSQL 和 Redis 环境执行完整 CTest。既有 `verify_unified_camera_pipeline.ps1` 能创建一次性依赖并覆盖数据库、Redis、回调、Worker 重启、RTSP 重连和运行时回滚基础能力。

车辆 Repository、HTTP 与 Callback 测试必须实际运行，不能保留 Skip 77。

### 3. 双路性能

使用两个授权且冻结的 1080p 输入，配置 `target_infer_fps=8` 或更高：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\benchmark_dual_camera_pipeline.ps1 `
  -CameraIds gate_01,gate_02 `
  -TargetInferFps 8 `
  -DurationMinutes 30
```

脚本记录每路吞吐、推理延迟、Hub 帧龄、替换/失败/旧结果计数、CPU、内存、GPU 和磁盘水位。只有 Vehicle Detection 已接入统一运行时后，该结果才能作为车辆 Detection FPS 证据。

### 4. 八小时稳定性

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\soak_camera_frame_feature.ps1 `
  -CameraId gate_01 `
  -DurationMinutes 480 `
  -PollSeconds 5 `
  -RequireCameraPipelineSubscriber `
  -CaptureHostTelemetry
```

正式 Evidence 还必须包含第二路状态、车辆队列、最终事件幂等和 Run generation 检查。短时冒烟不能替代 8 小时门禁。

### 5. 故障与回滚

在隔离验收环境使用现有脚本完成：

- `exercise_rtsp_reconnect.ps1`
- `exercise_worker_restart.ps1`
- `exercise_dead_letter_replay.ps1`
- `backup_runtime.ps1` 与 `restore_runtime.ps1`
- `exercise_runtime_mode_rollback.ps1`

PostgreSQL 和 Redis 故障只能操作一次性容器或明确的验收实例，禁止停止共享/生产服务。每个结果应保存为 `reports/m5/<release_id>/faults/<name>.json` 并在 Evidence 中引用项目相对路径。

## 发布包

Evidence 严格通过后执行：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\package_vehicle_release.ps1 `
  -EvidencePath .\reports\m5\vehicle_v1_0_0\release_evidence.json `
  -BuildDir .\out\build\backend-Release
```

打包器会再次执行 Evidence 与模型审计，要求干净工作区和完全匹配的 Git Commit，并收集：

- Server、Worker 和相邻运行时 DLL。
- Server、Worker、Camera、车辆、标签和验收策略配置。
- PostgreSQL v1–v4 迁移。
- Model Registry、ONNX、目标机 Engine、模型卡和指标。
- 启停、备份、恢复、模型/精度/Evidence 验证工具。
- Release Evidence 和逐文件 SHA256 Manifest。

输出 ZIP 只能用于本项目的非商业学习与技术验证。TensorRT Engine 标记为 `target_host_only`，不得在不同 TensorRT/CUDA/GPU 环境间假定可移植。

## 回滚

发布前保留上一版本 ZIP 和数据库备份。回滚时：

1. 停止 Server 和 Worker。
2. 保存失败版本日志、Evidence 和数据库备份。
3. 恢复上一版本数据库备份。
4. 恢复上一版本二进制、配置和 Engine。
5. 启动服务并验证 `/health`、`/ready`、Camera 状态、模型状态和 Callback Outbox。
6. 运行固定回放冒烟并确认没有旧 Run 结果进入新 generation。

没有完成实际恢复和重新就绪验证，不能把 `rollback_drill` 标为 `passed`。
