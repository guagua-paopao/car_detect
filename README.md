# Car Detect

Car Detect（项目代号 VCAS，Vehicle Cascade Analytics Service）是一个面向固定监控摄像头的车辆二级属性分析学习项目。

项目计划从 RTSP 视频中完成车辆检测、摄像头内跟踪、车辆裁剪质量门控、车身类型与颜色识别、轨迹级多帧融合，并生成可查询、可回调、可追溯的车辆经过事件。

## 当前状态

当前仓库处于 **M4 存储与 API 已实现、等待集成补验阶段**；M2R 的真实模型交付项继续等待云端训练产物。

- 基线来源：`guagua-paopao/vision_project`
- 基线提交：`c9f5000`
- 导入方式：压缩为 `car_detect/main` 的单个初始提交
- 复用范围：RTSP FrameHub、Camera Pipeline、有界推理池、任务控制、PostgreSQL、可靠回调和可观测性
- 迁移原则：车辆业务使用独立 Processor 和数据模型，不在人物处理器中继续堆叠分支
- 当前实现：M3 级联运行时、M2R TensorRT Adapter/交付门禁、M4 车辆事件存储与只读 API
- 暂缓验收：一次性 PostgreSQL 集成测试，以及真实 ONNX/Engine、SHA256、模型卡和精度回归

基线中的人物、Pose 和 People Flow 代码暂时作为可构建参考保留；车辆模块完成等价替换后再逐步删除。历史阶段日志、旧性能报告、旧 Postman 集合和机器相关 TensorRT Engine 不进入新仓库。

## MVP 范围

包含：

- 1 至 2 路 1080p RTSP 长期运行、断流重连、启停与状态查询
- 车辆检测和摄像头内跟踪
- 高质量车辆裁剪选择和独立属性微批推理
- 车身类型、粗粒度颜色和 `unknown` 降级
- 轨迹级多帧融合、最佳快照和 `vehicle_passage` 事件
- PostgreSQL 持久化、历史查询、HTTP 回调和可观测性
- ONNX 交付及目标 Windows 机器上的 TensorRT 构建与验证

首版不包含：

- 品牌、车系、年款和车牌 OCR
- 跨摄像头 ReID 或全局车辆身份
- 真实速度、世界坐标标定和多 Worker 调度
- 自动训练、自动发布或在线学习

## 目标架构

```text
RTSP Camera
  -> Shared Camera FrameHub
  -> Per-camera Pipeline
  -> Vehicle Detection Pool
  -> Vehicle Tracker
  -> Crop Quality Gate
  -> Vehicle Attribute Batch Pool
  -> Track-level Fusion
  -> PostgreSQL / Snapshot / Callback Outbox
```

一级检测池保留“每个摄像头只保留最新待处理帧”的背压语义。二级属性池使用全局有界队列、按轨迹去重和质量优先策略，避免属性分类拖慢检测主链路。

## 目录

```text
api/          API Schema 和模型接口
config/       Server、Worker 与摄像头配置
db/           PostgreSQL Schema 和迁移
deploy/       本地依赖编排
docs/         当前有效的架构、接口、运维和基线记录
include/      C++ 头文件
scripts/      构建、测试、运维和验收脚本
src/          业务、运行时与服务端实现
tests/        单元、契约和集成测试
tools/        Mock 与辅助验证工具
web/          管理端
qt_client/    Qt 管理客户端
```

模型和运行产物不提交到 Git：

- `*.engine`、`*.plan`
- `*.onnx`、`*.pt`、`*.pth`
- 本地构建目录、运行目录、报告和数据库
- 密钥、DSN 和本机专用配置

## 基线文档

- [项目范围](docs/PROJECT_SCOPE.md)
- [基线导入记录](docs/BASELINE_IMPORT.md)
- [需求与代码追溯矩阵](docs/TRACEABILITY.md)
- [架构决策记录](docs/DECISIONS.md)
- [阶段开发记录](docs/development/README.md)
- [数据标注规范](docs/data/ANNOTATION_GUIDE.md)
- [数据来源与许可台账](docs/data/LICENSE_LEDGER.md)
- [车辆模型接口](docs/models/MODEL_INTERFACE.md)
- [模型注册表](models/manifests/model_registry.v1.json)
- [模型交付回补状态](docs/models/MODEL_DELIVERY_STATUS.md)
- [车辆级联运行时](docs/runtime/VEHICLE_CASCADE_RUNTIME.md)
- [车辆事件 API](docs/api/VEHICLE_API.md)
- [车辆事件存储设计](docs/storage/VEHICLE_EVENT_STORAGE.md)
- [现有架构参考](docs/ARCHITECTURE.md)
- [Camera API](docs/CAMERA_FRAME_TASK_API.md)
- [部署、监控、故障与回滚](docs/CAMERA_FRAME_TASK_OPERATIONS.md)
- [Camera ID 与 PostgreSQL 设计](docs/CAMERA_INSTANCE_POSTGRESQL_DESIGN.md)

## 开发顺序

1. M0：冻结基线提交、配置、测试输入和性能口径。
2. M1：建立车辆标签、数据许可台账和试标规范。
3. M2：实现检测 Runner、属性 Runner 和模型注册表。
4. M3：实现车辆跟踪、裁剪质量门控和轨迹级融合。
5. M4：实现车辆结果表、事件、快照与 API。
6. M5：完成目标机 TensorRT、双路性能、8 小时稳定性和回滚验收。

每个阶段必须同步更新阶段记录、追溯矩阵、接口 Schema、配置版本和验证证据。破坏性接口变更必须升版，不能覆盖已经归档的 Schema。

## 构建环境

基线验证环境为 Windows x64、MSVC、Ninja、CUDA、TensorRT、OpenCV、Redis、PostgreSQL、Qt 和 vcpkg。依赖版本及路径应通过本机 CMake Preset 或脚本参数提供，不提交本机绝对路径。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_backend.ps1 `
  -VcpkgRoot D:\vcpkg `
  -CudaRoot D:\CUDA `
  -TensorRtRoot D:\TensorRT `
  -OpenCvDir D:\opencv\build\x64\vc16\lib
```

完整测试入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test_all.ps1
```

## 使用限制

本项目用于非商业学习和技术验证。数据集、训练框架、代码、预训练权重及派生模型必须分别遵守其许可证和用途限制；任何商业用途需要重新完成许可审查。
