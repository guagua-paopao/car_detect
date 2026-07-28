# VCAS M2 模型接口

本文件冻结一级车辆检测与二级属性分类之间的软件边界。M2 只定义接口、模型元数据和交付门禁，不宣称模型已经训练、导出或部署。

## 1. 核心文件

- C++ 接口：`include/server/vehicle_model_contract.h`
- C++ 契约实现：`src/server/vehicle_model_contract.cpp`
- 模型注册表：`models/manifests/model_registry.v1.json`
- 注册表 Schema：`api/schemas/model_registry.v1.schema.json`
- 注册表校验器：`tools/validate_model_registry.py`

## 2. Detection Runner

接口：`IVehicleDetectionRunner`

输入 `VehicleDetectionRequest` 必须携带：

- 图像内存视图；
- `camera_id` 和不可变 `run_id`；
- `run_generation`；
- `frame_sequence` 和采集时间。

输出 `VehicleDetectionResult` 必须回传：

- `artifact_id` 和 `labels_version`；
- 原始 `run_generation`、摄像头、Run 和帧序号；
- 归一化 `xyxy` 车辆框、一级类别和置信度；
- 推理耗时。

M3 运行时必须在接收结果时再次核对 `run_generation`，禁止旧 Run 的异步结果进入新轨迹。

## 3. Attribute Runner

接口：`IVehicleAttributeRunner`

`inferBatch` 接受多个 `VehicleAttributeCrop`，每个裁剪必须携带：

- 摄像头、Run 与 `run_generation`；
- `track_id` 和单调递增的 `crop_sequence`；
- 裁剪质量分数；
- 图像内存视图。

每个 `VehicleAttributeResult` 独立输出车身类型和颜色，不允许其中一个头的稳定状态强迫另一个头分类。Runner 输出单次裁剪观察，轨迹级投票、`unknown` 判定和最终事件发布属于 M3 级联运行时职责。

## 4. 模型注册表

注册表固定两个角色：

| 角色 | Artifact | 输入 | 输出 | 初始状态 |
|---|---|---|---|---|
| `detection` | `vehicle-det-v1` | 960×960 RGB NCHW | `vehicle_detections` | `planned` |
| `attributes` | `vehicle-attr-v1` | 224×224 RGB NCHW | `body_type`、`color` | `planned` |

合法交付状态：

1. `planned`：只有接口和目标路径，不允许填写虚假 SHA256。
2. `onnx_validated`：必须有 ONNX SHA256、数据版本、训练 Run、代码 Commit、模型卡和指标路径。
3. `engine_validated`：在 ONNX 证据基础上增加目标机 Engine SHA256。
4. `deployed`：已经通过目标机精度和性能门禁。
5. `blocked`：许可、精度、兼容性或证据存在问题，禁止装载。

TensorRT Engine 只在目标 Windows 主机生成。跨环境交付边界是 ONNX，不把云端或其他 GPU 生成的 Engine 视为可移植产物。

## 5. M3 集成规则

- 一级检测池只能依赖 `IVehicleDetectionRunner`。
- 二级属性池只能依赖 `IVehicleAttributeRunner`。
- 两个池分别初始化和释放，不共享有状态 Runner。
- 属性池可跨摄像头微批，但输出必须按 `run_generation + track_id + crop_sequence` 回配。
- Registry 和运行时配置不一致时启动失败，不使用静默默认值。
- 状态不是 `engine_validated` 或 `deployed` 的 Artifact 不得进入正式 TensorRT Worker。

## 6. 当前限制

- 没有真实 ONNX、TensorRT Engine、模型卡或指标文件。
- 没有实现具体 TensorRT Detection/Attribute Adapter。
- 没有实现属性微批队列、轨迹去重和结果融合。
- 上述内容分别由训练阶段和 M3 级联运行时完成。
