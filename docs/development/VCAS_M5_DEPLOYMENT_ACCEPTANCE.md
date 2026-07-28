# VCAS M5 部署与验收

## 状态

- 阶段：M5
- 状态：implemented_awaiting_acceptance
- 开始日期：2026-07-28
- 实现日期：2026-07-28
- 基线分支：`agent/vcas-m4-storage-api`
- 基线提交：`0eb05c106341132dfe7fa2a0964a07bc7a76d0d4`
- 当前分支：`agent/vcas-m5-deployment-acceptance`
- 阶段实现提交：本文件所在 M5 提交，以 Git/PR 记录为准
- 阶段完成提交：不适用；等待真实模型与目标机验收

## 完成范围

- 冻结学习版发布验收策略：双路 1080p、每路 Detection ≥8 FPS、延迟、8 小时稳定、幂等、故障和回滚门禁。
- 新增版本化 Release Evidence Schema 和明确为 `pending` 的示例。
- 新增严格 Evidence 验证器；默认失败关闭，只有规划记录可显式使用 `--allow-pending`。
- 新增候选 Evidence 生成脚本，自动冻结 Commit、脏状态、配置/Schema/Registry SHA256 和模型审计输出。
- 新增 Windows 发布打包器；严格复验 Evidence、模型资产、干净工作区和 Commit 后才生成逐文件 SHA256 ZIP。
- 复用既有双路性能、长稳、RTSP 重连、Worker 重启、Callback dead-letter、备份恢复和运行时回滚脚本。
- 增加 M5 正反契约测试与 PowerShell 解析门禁。

## 版本化接口

- 策略：`config/vehicle_release_acceptance.v1.json`
- Schema：`api/schemas/vehicle_release_evidence.v1.schema.json`
- Pending 示例：`api/examples/vehicle_release_evidence.pending.v1.json`
- 验证器：`tools/validate_vehicle_release_evidence.py`
- 候选记录：`scripts/new_vehicle_release_evidence.ps1`
- 发布包：`scripts/package_vehicle_release.ps1`
- 操作说明：`docs/operations/VEHICLE_RELEASE_ACCEPTANCE.md`

## 已完成验证

- [x] Pending Evidence 必须被严格门禁拒绝。
- [x] 正好达到 8 FPS、1500 ms、2000 ms 和 8 小时的边界证据通过。
- [x] 7.99 FPS、旧 Run 串入、缺模型和不完整回滚证据被拒绝。
- [x] Release Artifact 路径和 SHA256 使用失败关闭规则。
- [x] PowerShell 脚本可解析。
- [x] 候选生成器可在模型缺失时生成 `pending` 记录，且不会声称发布就绪。

## 待真实环境验收

- [ ] Detection/Attribute ONNX、Engine、SHA256、模型卡和精度回归。
- [ ] Vehicle Detection/Attribute 真实 Worker 集成和硬件冒烟。
- [ ] 两路 1080p、每路 Detection ≥8 FPS 的 30 分钟测试。
- [ ] 属性稳定和事件持久化 P95 测试。
- [ ] 8 小时双路稳定性与显存趋势。
- [ ] PostgreSQL、Redis、RTSP、Callback 和 Worker 故障演练。
- [ ] 数据库备份恢复及上一版本二进制/配置/Engine 回滚演练。
- [ ] 严格 Evidence 通过后的 Windows Release ZIP。

这些项目在真实模型、冻结测试集和隔离验收环境到位前保持未完成。

## 退出条件

M5 只有在严格 Evidence 验证器无 `BLOCKED` 项、发布 ZIP 可在目标 Windows 主机重复部署、回滚后重新就绪，并且全部证据路径和 SHA256 可追溯时才能改为 `completed`。

## 回滚

M5 当前只增加策略、证据、验证和打包层，不改变服务默认运行路径。回滚本阶段不会影响 M4 数据库/API 或 M3 级联核心。正式发布后的回滚必须按操作文档恢复数据库、二进制、配置和 Engine 的上一组一致版本。
