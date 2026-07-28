# Vehicle Event Storage

M4 使用 PostgreSQL Schema v4。迁移文件为 `db/postgresql/004_vehicle_events.sql`，Repository 的内联初始化 Schema 与该迁移保持一致。

## 表和职责

| 表 | 职责 | 关键约束 |
|---|---|---|
| `vision_events` | 所有可回调视觉事件的不可变父记录和冻结 JSON | `event_id` 主键 |
| `security_alert_events` | 既有算法告警的类型化投影 | `event_id` 外键到 `vision_events` |
| `vehicle_track_results` | 每条完成车辆轨迹的可查询类型化投影 | `event_id` 主键；`UNIQUE(run_id, track_id)` |
| `vehicle_attribute_observations` | 可选的属性调试观测 | `UNIQUE(run_id, track_id, crop_sequence)`；按 `expires_at_ms` 清理 |
| `callback_outbox` | 可靠回调状态机 | `event_id` 唯一且外键到 `vision_events` |

迁移会先把既有 `security_alert_events` 回填到 `vision_events`，再把 Callback Outbox 的外键从告警表迁移到通用事件表，因此原有告警和回调审计记录不会丢失。

## 原子发布

`CameraTaskRepository::publishVehicleEvent` 在一个事务内：

1. 写入 `vision_events`，冻结 `vehicle_passage` 回调 JSON。
2. 写入 `vehicle_track_results` 类型化查询投影。
3. 摄像头配置了 Callback Profile 时，写入 `callback_outbox`。
4. 任一步失败则回滚全部记录。

`VehicleEventPublisher` 把 M3 的 `VehicleTrackSnapshot` 和可选 `VehicleTrackAttributeSnapshot` 映射为持久化记录。事件 ID 由 Camera、Run、generation 和 Track ID 的 SHA256 前缀确定；重复发布读取已存在投影并返回幂等成功。数据库的 `(run_id, track_id)` 唯一约束提供第二层防重。

回调 Worker 通过 `CameraTaskRepository::getVisionEvent` 读取冻结 JSON。重试不会重新序列化当前模型、配置或标签状态，因此签名 Body 与首次发布保持一致。

## 数据校验

发布前检查：

- `task_id` 必须等于 `camera_id`，Run、Camera、事件和 Artifact 标识符必须在允许范围内。
- `first_seen_at_ms <= last_seen_at_ms <= occurred_at_ms <= created_at_ms`。
- 车辆类别、车身类别、颜色和结束原因必须属于冻结枚举。
- 所有置信度和 `crop_quality` 必须在 `[0,1]`。
- 稳定属性至少使用一个样本。
- 快照只能保存输出根目录内的项目相对路径。

数据库外键继续保证 Camera 和 Run 存在。

## 调试观测保留

`VehicleEventPublisher::persistObservation` 为每个 `(Run, Track, crop_sequence)` 生成确定性 Observation ID。保留天数被限制在 1–365 天，默认值由 `vehicle_analytics.observation_retention_days` 提供。

`CameraTaskRepository::deleteExpiredVehicleAttributeObservations` 以有界批次删除过期记录。该清理不删除最终车辆事件、快照元数据或 Callback Outbox。

## 快照

数据库只保存 `snapshot_relative_path`，实际 JPEG 位于 `camera_tasks.output_dir` 下。HTTP 层使用既有安全路径解析器阻止绝对路径和目录穿越，并在返回前校验 JPEG 头尾标记。

## 运维与回滚

升级前应备份数据库。Schema v4 是向前迁移：保留既有告警表并扩展通用事件父表。应用回滚到不认识 Schema v4 的版本前，应先确认旧应用是否接受更高 Schema 版本；不应直接删除 `vision_events`，因为 Callback Outbox 已引用它。

如必须数据库回退，应在维护窗口：

1. 停止 Server 和 Worker。
2. 确认没有 `vehicle_passage` Outbox 待发送。
3. 从备份恢复 Schema v3 数据库。
4. 回滚应用版本并重新启动。

不要通过手工删除 v4 表来代替备份恢复。
