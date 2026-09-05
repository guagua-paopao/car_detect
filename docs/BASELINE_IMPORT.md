# 基线导入记录

## 来源

- 上游仓库：`https://github.com/guagua-paopao/vision_project.git`
- 上游分支：`agent/unified-camera-r8`
- 上游提交：`c9f5000`
- 导入日期：`2026-07-28`
- `car_detect/main` 根提交：`44c29fc84c5aaf49980b1ebb2319858b566902cb`
- 根 Tree：`ea2ebcdb96f95ee0cc12bbe7a121775d6aad2c44`

`car_detect` 使用独立的单提交历史。上游仓库保留为本地 `upstream` 远程，仅用于追溯和对照，不作为新项目的发布目标。

## 导入时保留

- Shared Camera FrameHub 和 RTSP 重连
- Camera Pipeline 与最新帧优先机制
- 固定大小、有界的 Camera Inference Pool
- Camera CRUD、RunSpec、任务租约和代际抑制
- PostgreSQL Repository 与迁移框架
- Callback Outbox、重试、dead-letter 和 HMAC
- Web 控制面、可观测性和运维脚本
- 现有契约、单元和集成测试

## 导入时移除

- 旧项目 M0-M11、P0-P6 和 Unified Camera 阶段日志
- 旧人物项目计划、重构计划、验收和性能对比报告
- 旧 Postman 集合
- 已序列化的 Pose TensorRT Engine 及其校验清单
- 重复的旧项目范围 DOCX

本地的 `out/`、`runtime/`、`reports/` 和 `vcpkg_installed/` 属于忽略的构建或运行资产，不进入 Git 历史，也不在导入过程中主动删除。

## 兼容代码清理结果

基线导入时曾暂时保留人物、姿态估计和人流统计源码，以便验证车辆替代链路。车辆链路达到测试覆盖后，相关构建目标、API、配置、数据库对象、脚本和测试均已删除。

清理过程遵守了以下顺序：

1. 新增独立车辆类型和接口。
2. 新增 Vehicle Detection Runner 和 Vehicle Attribute Runner。
3. 新增 Vehicle Cascade Processor 和独立属性池。
4. 新增车辆数据库、API 和测试。
5. 车辆链路达到基线测试覆盖后，删除被替代的人物专用目标和资产。

车辆逻辑保持为独立实现，不与已删除的人物处理分支混用。
