# 需求与代码追溯矩阵

本矩阵把范围文档中的要求映射到配置、接口、实现位置和验证证据。状态只允许使用 `planned`、`in_progress`、`verified` 或 `blocked`。

| ID | 要求 | 当前状态 | 配置或接口 | 实现位置 | 验证证据 |
|---|---|---|---|---|---|
| VCAS-FR-001 | 每个 Camera Profile 最多一个 RTSP Reader | verified | `config/cameras.yaml` | `shared_camera_frame_hub.*` | `shared_camera_frame_hub_test` |
| VCAS-FR-002 | 一级车辆检测采用最新帧优先的有界队列 | planned | `config/vehicle_analytics.yaml` | 计划：Vehicle Detection Pool | 计划：车辆检测池测试 |
| VCAS-FR-003 | 摄像头内车辆跟踪和确认状态机 | verified | `min_confirm_hits`、`track_timeout_ms`、IoU 阈值 | `VehicleTracker` | `vehicle_cascade_runtime_test` |
| VCAS-FR-004 | 只向二级模型提交高质量车辆裁剪 | verified | 尺寸、清晰度、曝光、遮挡、截断门槛 | `VehicleCropQualityGate` | 合格与拒绝路径测试 |
| VCAS-FR-005 | 二级属性池独立、有界并按 Track 去重 | in_progress | `queues.attribute_*` | `VehicleAttributeCandidateQueue`；真实 Runner 池待接入 | 背压、替换、淘汰测试 |
| VCAS-FR-006 | 车身类型和颜色分别进行轨迹级融合 | verified | `attribute_vote_samples`、阈值 | `VehicleTrackAttributeAggregator` | 独立稳定与 unknown 测试 |
| VCAS-FR-007 | 每条轨迹至多生成一个车辆事件 | in_progress | `vehicle_event.v1.schema.json` | 计划：Vehicle Event Publisher | `vehicle_contract_test.py` |
| VCAS-FR-008 | 车辆事件可查询、可回调、可追溯 | in_progress | `vehicle_event.v1.schema.json` | 复用 Repository 与 Callback Outbox | `vehicle_contract_test.py` |
| VCAS-NFR-001 | 双路 1080p，每路检测平均不低于 8 FPS | planned | `detection_fps` | 计划：容量基准 | M0 固定回放待冻结 |
| VCAS-NFR-002 | 8 小时无崩溃、显存持续增长或旧 Run 串入 | planned | Run generation | 复用代际抑制与心跳 | 计划：8 小时稳定性报告 |
| VCAS-NFR-003 | 数据、模型、配置和代码版本可追溯 | in_progress | `config_version`、`model_versions` | Schema 与阶段记录 | `vehicle_contract_test.py` |
| VCAS-NFR-004 | 不在 Git 中保存机器专用模型和运行产物 | verified | `.gitignore` | 仓库策略 | M0 tracked-file audit |
| VCAS-DATA-001 | 标签映射、别名和 unknown 策略必须版本化 | verified | `vehicle_labels.v1.json` | M1 数据契约 | `dataset_manifest_contract_test.py` |
| VCAS-DATA-002 | 未批准来源不得进入训练、验证或测试 | in_progress | Dataset Manifest `sources` | Manifest Validator | 许可门禁负向测试 |
| VCAS-DATA-003 | 摄像头、视频和轨迹分组不得跨 Split | verified | `split_policy.group_keys` | Manifest Validator | 泄漏负向测试 |
| VCAS-DATA-004 | 样本文件、来源和标签版本必须可追溯 | verified | Dataset Manifest v1 | Manifest Validator | Manifest 示例与契约测试 |
| VCAS-MODEL-001 | 一级检测与二级属性使用独立强类型 Runner | verified | `vehicle_model_contract.h` | Detection/Attribute Runner 接口 | C++ `vehicle_model_contract_test` |
| VCAS-MODEL-002 | 模型角色、输入、输出、精度和目标平台必须版本化 | verified | Model Registry v1 | Registry Validator | `model_registry_contract_test.py` |
| VCAS-MODEL-003 | 异步模型结果保留 Artifact、标签、Run generation 和序列上下文 | verified | Runner Request/Result | M2 C++ 契约 | C++ Fake Runner 往返测试 |
| VCAS-MODEL-004 | 未完成校验和及来源证据的模型不得声明已部署 | verified | `delivery_status`、SHA256、provenance | Registry Validator | 虚假部署状态负向测试 |
| VCAS-MODEL-005 | ONNX 跨环境交付，TensorRT Engine 在目标 Windows 主机构建 | in_progress | Registry `target`、文件字段 | 计划：TensorRT Adapter | 计划：ONNX/TRT 精度回归 |
| VCAS-RUNTIME-001 | 检测和属性结果必须抑制旧 Run 与旧序列 | verified | `run_generation`、frame/crop sequence | M3 Tracker 与 Aggregator | 旧代际和旧序列负向测试 |
| VCAS-RUNTIME-002 | 异步属性裁剪必须保证像素生命周期安全 | verified | `shared_ptr<const OwnedImage>` | M2/M3 图像契约 | 拥有型裁剪复制与批次测试 |
| VCAS-RUNTIME-003 | 属性过载优先保留高质量、更新的 Track 候选 | verified | Queue Metrics | M3 Attribute Queue | 容量淘汰与质量排序测试 |

## 更新规则

1. 开始实现时，将对应项改为 `in_progress`。
2. 只有实现、自动测试和验收证据全部存在时才能改为 `verified`。
3. Schema、数据库或配置字段变化时，同一提交必须更新本矩阵。
4. 阶段结束时，将测试日志和关键指标摘要写入对应阶段记录。
