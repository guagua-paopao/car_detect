# VCAS M0 基线与接口冻结

## 状态

- 阶段：M0
- 状态：in_progress
- 开始日期：2026-07-28
- 基准分支：`main`
- 基准提交：`44c29fc84c5aaf49980b1ebb2319858b566902cb`
- 上游参考提交：`vision_project@c9f5000`

## 目标

1. 建立唯一、可复现的代码基线。
2. 冻结车辆 MVP 标签、配置结构和 `vehicle_passage` v1 事件接口。
3. 建立需求、代码、接口和测试的追溯入口。
4. 明确本阶段不改动高耦合人物/Pose 运行时。

## 本阶段产物

- `docs/DECISIONS.md`
- `docs/TRACEABILITY.md`
- `config/vehicle_analytics.yaml`
- `api/schemas/vehicle_event.v1.schema.json`
- `api/examples/vehicle_passage.v1.json`
- `tests/vehicle_contract_test.py`

## 基线验证

基线导入时执行：

- Python Qt/API 契约测试：通过
- Web Admin 契约测试：通过
- PowerShell 脚本语法检查：通过
- Markdown 相对链接检查：通过
- C++ 测试：11 个通过；11 个因未提供隔离 PostgreSQL/Redis 测试环境按设计跳过

全新 CMake 配置在当前自动化账户中被 MSVC `c1.dll` 执行环境阻断。该问题记录为 M0 环境阻塞项，不能用旧构建目录替代重新配置证据。

## 待冻结证据

- [ ] 目标 Windows 硬件、驱动、CUDA、TensorRT 和 OpenCV 版本
- [ ] 两路固定回放或固定 RTSP 输入清单及 SHA256
- [ ] `worker.yaml` 和车辆配置快照
- [ ] 唯一的双路 FPS、P50/P95 延迟、CPU、GPU 和显存报告
- [ ] 测试 PostgreSQL/Redis 环境下的全部集成测试结果

## M0 退出条件

- [ ] 所有本阶段产物已合并到 `main`
- [ ] `vehicle_contract_test.py` 进入 GitHub Actions 并通过
- [ ] 目标机可从干净构建目录完成 CMake 配置
- [ ] 固定输入、配置、硬件和性能采样脚本全部可追溯
- [ ] 追溯矩阵中 M0 项状态与证据一致

## 回滚

M0 只新增文档、配置和接口契约。发生问题时可回滚本阶段提交；基线根提交和 `upstream` 引用保持不变。
