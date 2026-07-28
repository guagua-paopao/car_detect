# VCAS M4 存储与 API

## 状态

- 阶段：M4
- 状态：implemented_pending_integration
- 开始日期：2026-07-28
- 实现日期：2026-07-28
- 基线分支：`agent/vcas-model-delivery-return`
- 基线提交：`898bf4040432d1d22ae01d6f77934425dec57e77`
- 当前分支：`agent/vcas-m4-storage-api`
- 阶段实现提交：本文件所在 M4 提交，以 Git/PR 记录为准
- 阶段完成提交：不适用；等待 PostgreSQL 与真实模型集成补验

## 完成范围

- 新增通用不可变 `vision_events` 父表、车辆轨迹结果表和属性调试观测表。
- 迁移既有算法告警，并把可靠 Callback Outbox 扩展为支持所有视觉事件。
- M3 轨迹和融合属性到持久化记录的强类型发布接口。
- 基于 Camera、Run generation 和 Track 的确定性事件 ID，以及数据库唯一约束的双层幂等。
- 车辆实时、历史、详情、JPEG 快照和模型交付状态 API。
- 快照相对路径约束、目录穿越防护和 JPEG 完整性检查。
- 属性调试观测的有界保留和批量清理接口。
- 运维 JSON 和 Prometheus 指标增加视觉事件、车辆事件与属性观测计数。

## 配置、Schema 与版本

- PostgreSQL Schema：v4，迁移 `db/postgresql/004_vehicle_events.sql`。
- Vehicle Event Schema：`api/schemas/vehicle_event.v1.schema.json`，版本 1.0。
- 标签版本：由 `labels_version` 随每条事件冻结。
- 模型版本：Detection 和 Attribute Artifact ID 随每条事件冻结。
- 配置版本：`config_version` 随每条事件冻结。
- 新增 Server 配置：`vehicle_analytics.enabled`、`config_path`、`model_registry_path` 和 `observation_retention_days`。

## 接口变化

- 新增 `VisionEventRecord`、`VehicleTrackResultRecord`、`VehicleAttributeObservationRecord` 和 `VehicleRealtimeTrackRecord`。
- `CameraTaskRepository` 新增通用事件读取、车辆事件原子发布与查询、属性观测写入和过期清理接口。
- 新增 `VehicleEventPublisher`，连接 M3 轨迹快照、融合属性与 M4 Repository。
- `CallbackDeliveryWorker` 改为读取不可变 `VisionEventRecord::payload_json`，保留既有签名、重试和死信语义。
- `CameraTaskHttpController` 新增实时 Reader 注入点和五个车辆/模型只读接口。

详细接口见 [Vehicle Event API](../api/VEHICLE_API.md)，数据设计见 [Vehicle Event Storage](../storage/VEHICLE_EVENT_STORAGE.md)。

## 已完成验证

- [x] M4 C++ 目标在 Windows Release 配置下编译、链接。
- [x] App Config 运行时测试通过。
- [x] M3 级联运行时回归通过。
- [x] M0–M4 Python 源码与 Schema 契约回归通过。
- [x] API 契约覆盖实时不可用降级、历史筛选、详情、快照和计划态模型。
- [x] Repository 集成测试覆盖原子发布、同 Run/Track 幂等、观测保留、统计和 Outbox。

## 暂缓验证

- [ ] Repository、Callback 和 HTTP 的 PostgreSQL 集成测试：测试可执行文件已成功构建，但当前环境没有 `YOLO11_TEST_POSTGRES_DSN` 指向一次性测试库，因此 CTest 按约定返回 Skip 77。
- [ ] Detection/Attribute 真实推理、ONNX/Engine、SHA256、模型卡和精度回归：模型仍在云服务器训练，继续由 M2R 保持 `awaiting_assets`。
- [ ] 车辆端到端真实视频验收：需在真实模型和一次性测试库可用后，与上述测试一起执行。

暂缓项不改写为通过，也不以随机权重、空 Engine 或生产数据库替代。

## 退出条件

- [x] 数据迁移和 Repository 接口完成。
- [x] 车辆事件原子发布与可靠 Outbox 完成。
- [x] 实时、历史、详情、快照和模型状态 API 契约完成。
- [x] 阶段接口、存储和回滚文档完成。
- [ ] PostgreSQL 集成测试在一次性数据库实际通过。
- [ ] 真实模型端到端测试在训练资产交付后通过。

## 已知限制

- 实时 API 需要 Worker 将 `VehicleRealtimeSnapshotReader` 接入运行中的 `VehicleCascadeRuntime`；未接入时明确返回 `available=false`。
- 模型状态接口只报告 Registry 声明，不替代 TensorRT Adapter 的 SHA256 加载门禁。
- 快照生成由运行时负责；M4 只持久化相对路径并安全提供已有 JPEG。
- 本阶段不替换既有人物/Pose 演示处理链路。

## 回滚

代码可回滚到基线提交，但数据库已迁移到 Schema v4 时应优先使用升级前备份恢复，不应手工删除 `vision_events`。详细步骤见存储设计文档。
