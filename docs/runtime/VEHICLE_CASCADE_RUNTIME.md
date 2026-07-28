# VCAS M3 车辆级联运行时

M3 将 M2 的模型输入输出接口连接为可独立测试的摄像头内运行时核心。当前实现不依赖 OpenCV、CUDA 或 TensorRT，可在 CI 中验证调度与状态语义；真实模型 Adapter 在 M3 完成后回补。

## 1. 运行链路

1. `VehicleTracker` 接收同一 Camera、Run 和 generation 的有序检测结果。
2. 轨迹由 `TENTATIVE` 达到 `CONFIRMED` 后才允许提交属性裁剪。
3. `VehicleCropQualityGate` 依据尺寸、清晰度、曝光、遮挡和截断评估真实像素。
4. 通过门禁的裁剪复制为不可变 `OwnedImage`，避免异步队列悬空引用。
5. `VehicleAttributeCandidateQueue` 每个 Track 最多保留一个待处理裁剪，质量提升才替换。
6. 队列按质量优先形成跨 Track 微批。
7. `VehicleTrackAttributeAggregator` 对车身类型和颜色分别进行质量×置信度加权。
8. 两个属性独立达到票数和阈值后，轨迹进入 `ATTR_STABLE`。
9. 轨迹退出或 Run 停止时，上层调用 `finalizeTrack`；不稳定属性保持 `unknown`。

## 2. 轨迹状态

`NEW -> TENTATIVE -> CONFIRMED -> ATTR_COLLECTING -> ATTR_STABLE -> EXITED`

- `min_confirm_hits` 控制确认门槛。
- 匹配使用摄像头内归一化框 IoU；当前实现是确定性贪心匹配。
- `track_timeout_ms` 控制未匹配轨迹退出。
- `run_generation`、Camera、Run 或帧序号不一致的检测结果被拒绝。
- 当前匹配核心用于冻结状态和接口语义；后续可以在不改变 M3 外部接口的前提下替换为 ByteTrack 风格实现。

## 3. 质量门禁

默认门槛来自 `config/vehicle_analytics.yaml`：

- 最小宽度 96 px、最小高度 64 px；
- 最小归一化清晰度 0.02；
- 曝光范围 0.08～0.95；
- 最大遮挡比例 0.50；
- 默认拒绝截断裁剪。

`quality_score` 综合尺寸、水平梯度清晰度、曝光、可见比例和截断状态，范围为 `[0, 1]`。门禁失败会返回稳定的原因码，如 `crop_too_small`、`crop_too_blurry`、`exposure_out_of_range`、`too_occluded` 或 `truncated`。

## 4. 背压与去重

- 全局待处理上限：`attribute_max_pending`。
- 每 Track 待处理上限：1。
- 新裁剪序号必须严格递增。
- 同一 Track 只有质量更高的新裁剪可以替换旧候选。
- 队列已满时，低于当前最低质量的裁剪被丢弃；更高质量裁剪淘汰最低质量候选。
- 所有接收、替换、重复、过期、溢出丢弃和淘汰均有计数。

## 5. 轨迹级融合

- 默认至少 3 个同标签有效观察。
- 最多保留 5 个最高质量观察。
- 每票权重为 `quality_score × model_confidence`。
- 车身类型与颜色分别使用 `type_threshold` 和 `color_threshold`。
- `unknown` 观察计入总质量但不为任何具体标签投票，因此会降低覆盖率而不会被强制归类。
- Artifact 或标签版本在同一 Track 内变化时拒绝结果。

## 6. 接口文件

- `include/business/vehicle_cascade_runtime.h`
- `src/business/vehicle_cascade_runtime.cpp`
- `tests/vehicle_cascade_runtime_test.cpp`

M4 存储与 API 阶段将消费最终 `VehicleTrackAttributeSnapshot`，负责幂等事件、快照、查询和 Outbox；这些责任不进入 M3。
