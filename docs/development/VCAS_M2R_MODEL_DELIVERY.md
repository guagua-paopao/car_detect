# VCAS M2R 模型交付回补

## 状态

- 阶段：M2R
- 状态：awaiting_assets
- 开始日期：2026-07-28
- 基线分支：`agent/vcas-m3-cascade-runtime`
- 基线提交：`b46972ae350f8f7d3398a4edbd2014028364a640`
- 当前分支：`agent/vcas-model-delivery-return`
- Adapter 与交付门禁实现提交：`3974f1e`
- 阶段完成提交：不适用；等待真实模型与数据资产

## 已完成范围

- 实现 `TensorRtVehicleDetectionRunner`：TensorRT Engine 反序列化、动态输入 Shape、CUDA 传输、`enqueueV3`、YOLO 原始输出解码和分类别 NMS。
- 实现 `TensorRtVehicleAttributeRunner`：动态批量输入、双输出 Tensor、Softmax 与 Top-1。
- 在 Engine 反序列化前重新计算 SHA256，并与 Model Registry 记录比较。
- 只允许加载 `engine_validated` 或 `deployed` 状态的 Artifact。
- 增加基于真实 `trtexec` 的目标机 Engine 构建脚本。
- 增加 ONNX/Engine、SHA256、来源、模型卡和指标文件审计工具。
- 增加同一 Artifact、数据版本、标签版本和样本数量下的精度回归门禁，默认允许绝对下降不超过 `0.005`。

## 接口变化

- 新增 `TensorRtDetectionOptions`、`TensorRtAttributeOptions`。
- Adapter Options 通过 `artifact_root` 明确解析 Registry 中的项目相对路径，避免依赖服务进程恰好从仓库根目录启动。
- 新增 `TensorRtVehicleDetectionRunner`、`TensorRtVehicleAttributeRunner`。
- `ImageView` 和 `OwnedImage` 新增 `ImagePixelFormat`，明确 BGR/RGB 输入语义。
- M3 的拥有型车辆裁剪保留源图像像素格式。

## 已完成验证

- [x] MSVC 19.51、CUDA 13.3、TensorRT 10.16 和 OpenSSL 实际编译、链接。
- [x] Detection/Attribute Adapter 对计划态 Artifact 和缺失 Engine 执行失败关闭。
- [x] 源码门禁确认真实使用 TensorRT、CUDA 和 OpenSSL API。
- [x] 模型交付审计器准确报告当前 Registry 的缺失资产与来源字段。
- [x] 精度门禁正反例：下降 0.4 个百分点通过，下降 1 个百分点失败。
- [x] M0–M3 契约回归。

## 尚未完成及原因

- [ ] Detection/Attribute 真实 ONNX：仓库及目标机均未提供训练产物。
- [ ] Detection/Attribute 真实 TensorRT Engine：缺少可转换 ONNX。
- [ ] 真实 ONNX/Engine SHA256：文件不存在，不能生成可信哈希。
- [ ] 模型卡：缺少训练 Run、授权数据版本、训练代码提交和真实指标。
- [ ] 原始模型、ONNX、TensorRT 精度回归：缺少同一封闭测试集的逐样本结果。

这些项目必须继续保持未完成。随机网络、空 ONNX、虚构哈希或仅验证 Engine 可加载都不能替代真实模型交付与精度验收。

## 解阻输入

1. 经许可审核并冻结的数据集版本及测试集。
2. Detection/Attribute 训练 Run 或已批准的真实 ONNX。
3. 标签版本、训练代码 Commit 和原始框架指标。
4. 对同一测试集导出的原始、ONNX 和 TensorRT 逐样本结果。

获得输入后，按 `docs/models/MODEL_DELIVERY_STATUS.md` 的顺序构建 Engine、写入 SHA256、完成模型卡与精度回归，并把 Registry 状态升级为 `engine_validated`。

## 回滚

Adapter 作为独立库加入，未替换现有人物处理器或车辆运行时默认实现。回滚本阶段只需移除 Adapter 库、构建目标和模型交付工具，不影响 M3 级联核心。
