# VCAS M3 级联运行时

## 状态

- 阶段：M3
- 状态：completed
- 开始日期：2026-07-28
- 完成日期：2026-07-28
- 基线分支：`agent/vcas-m2-model-interfaces`
- 基线提交：`667e21973560a614dd618c8c3190fa2e44e8640b`
- 当前分支：`agent/vcas-m3-cascade-runtime`

## 完成范围

- 摄像头内车辆轨迹确认、IoU 匹配、超时退出和状态迁移。
- Camera、Run、generation 和帧序号校验。
- 基于真实像素的尺寸、清晰度、曝光、遮挡和截断门禁。
- 异步安全的不可变拥有型车辆裁剪。
- 全局有界、每 Track 去重、质量替换和质量优先微批队列。
- 车身类型与颜色独立的质量加权轨迹融合。
- `unknown` 降级、Artifact/标签版本一致性和 Crop 序列抑制。
- 组合式 `VehicleCascadeRuntime` 接口。

## 接口变化

- 新增 `VehicleTracker`、`VehicleCropQualityGate`。
- 新增 `VehicleAttributeCandidateQueue`、`VehicleTrackAttributeAggregator`。
- 新增 `VehicleCascadeRuntime`。
- `VehicleAttributeCrop` 从非拥有型 `ImageView` 改为 `shared_ptr<const OwnedImage>`，保证异步队列生命周期安全。
- 配置升版为 `vehicle-analytics-m3`。

## 自动验证

- 轨迹从 Tentative 确认、Track ID 保持、超时退出。
- 旧 generation 拒绝和帧序号单调性。
- 合格、过小、模糊/暗光和截断裁剪。
- 裁剪像素复制及拥有权。
- 同 Track 质量替换、旧序列拒绝、容量淘汰和质量批次排序。
- 车身类型稳定但颜色未稳定的独立融合场景。
- 颜色新增有效票后稳定，以及重复 Crop 序列拒绝。
- 从检测确认到裁剪、微批、属性回填、稳定和停止的组合测试。

## M3 退出条件

- [x] 车辆轨迹状态机与旧 Run 抑制完成
- [x] 质量门控和拥有型裁剪完成
- [x] 有界属性候选队列、每 Track 去重和质量优先完成
- [x] 类型/颜色独立融合与 `unknown` 策略完成
- [x] C++17 独立编译测试进入 CI
- [x] 配置、决策、追溯和运行时接口文档同步

## 后续回补

按项目顺序，M3 完成后返回 M2 模型交付缺口：

1. 真实 Detection TensorRT Adapter。
2. 真实 Attribute TensorRT Adapter。
3. 训练或取得经过许可审核的 ONNX。
4. 在当前 RTX 4080 Laptop GPU 上生成 Engine。
5. 写入 SHA256、模型卡、指标和训练/数据/代码来源。
6. 完成原始模型、ONNX 与 TensorRT 的精度回归。

当前仓库没有 Detection/Attribute ONNX 或训练权重，因此上述模型产物不能在 M3 提交中虚构。

## 回滚

M3 作为独立 `vehicle_cascade_core` 静态库加入，不替换现有人物处理器、Pose Runner 或 Camera Algorithm Processor。回滚 M3 不影响现有演示链路。
