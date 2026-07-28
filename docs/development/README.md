# VCAS 阶段开发记录

本目录只保存 `car_detect` 的有效阶段记录。每个阶段文档必须包含：

- 目标、范围和退出条件
- 起始及结束 Commit SHA
- 配置、Schema、数据库和模型版本
- 新增或改变的代码接口
- 自动测试与人工验收结果
- 已知限制、风险和回滚方式

## 阶段索引

| 阶段 | 状态 | 记录 | 主要产物 |
|---|---|---|---|
| M0 基线与接口冻结 | in_progress | [VCAS_M0_BASELINE.md](VCAS_M0_BASELINE.md) | 基线清单、事件 Schema、配置契约、追溯矩阵 |
| M1 数据规范 | in_progress | [VCAS_M1_DATA_SPEC.md](VCAS_M1_DATA_SPEC.md) | 标签映射、许可台账、Manifest 与试标规范 |
| M2 模型接口 | in_progress | [VCAS_M2_MODEL_INTERFACES.md](VCAS_M2_MODEL_INTERFACES.md) | Detection/Attribute Runner、模型注册表 |
| M3 级联运行时 | completed | [VCAS_M3_CASCADE_RUNTIME.md](VCAS_M3_CASCADE_RUNTIME.md) | 跟踪、门控、属性队列、轨迹融合 |
| M4 存储与 API | planned | 待创建 | 数据迁移、车辆事件 API、快照和回调 |
| M5 部署与验收 | planned | 待创建 | TensorRT、性能、稳定性、故障和回滚报告 |

旧 `vision_project` 的阶段日志不复制到此目录；需要追溯时使用 `upstream` 仓库及 `docs/BASELINE_IMPORT.md` 中记录的提交。
