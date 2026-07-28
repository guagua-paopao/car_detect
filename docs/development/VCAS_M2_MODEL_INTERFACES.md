# VCAS M2 模型接口

## 状态

- 阶段：M2
- 状态：in_progress
- 开始日期：2026-07-28
- 基线分支：`agent/vcas-m1-data-contracts`
- 基线提交：`926400052dbf94e3f3778736658992a52bd7d439`
- 当前分支：`agent/vcas-m2-model-interfaces`

## 目标

1. 将一级车辆检测和二级属性分类从原有 Pose Runner 中解耦。
2. 冻结 Detection/Attribute 强类型请求和结果。
3. 建立模型注册表、交付状态和校验和门禁。
4. 保留 Run generation、Track 和 Crop 序列上下文，支持后续异步双池。
5. 明确 ONNX 跨环境交付与目标 Windows 主机构建 TensorRT Engine 的边界。

## 新增接口

- `IVehicleDetectionRunner`
- `IVehicleAttributeRunner`
- `VehicleModelRegistry`
- `ModelArtifactDescriptor`
- `VehicleDetectionRequest` / `VehicleDetectionResult`
- `VehicleAttributeCrop` / `VehicleAttributeResult`
- Model Registry v1 Schema 和 Manifest

详细字段和集成规则见 `docs/models/MODEL_INTERFACE.md`。

## 已完成验证

- [x] C++17 独立编译并运行 Detection/Attribute Fake Runner 契约测试
- [x] Registry 正例校验
- [x] 重复 Role、越界路径、虚假部署状态、输出漂移和配置漂移负向测试
- [x] M0 车辆事件契约和 M1 数据契约回归
- [x] CI 增加 Python 注册表测试和 C++ 编译运行测试

## M2 退出条件

- [x] 强类型 Detection/Attribute Runner 接口完成
- [x] 内存模型注册表及 Artifact 校验完成
- [x] Model Registry v1 Schema、计划态 Manifest 和无依赖校验器完成
- [x] Runtime Config、事件示例和模型版本保持一致
- [ ] Detection TensorRT Adapter 完成并通过固定输入测试
- [ ] Attribute TensorRT Adapter 完成批量输入与双头输出测试
- [ ] 真实 ONNX 产物、SHA256、模型卡和指标记录完成
- [ ] 目标 Windows Engine 构建及 ONNX/TensorRT 精度回归完成

## 回滚

M2 新接口未替换现有 Pose Runner，也没有加载任何新模型文件。回滚本阶段不会影响现有四阶段人物演示；删除 M2 契约、注册表引用和对应测试即可恢复 M1 状态。
