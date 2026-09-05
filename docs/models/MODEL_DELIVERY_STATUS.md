# VCAS 模型交付回补状态

核验日期：2026-07-29

## 已完成

- `TensorRtVehicleDetectionRunner` 已实现真实 TensorRT Engine 反序列化、CUDA 输入输出、YOLO 原始输出解码和 NMS。
- `TensorRtVehicleAttributeRunner` 已实现动态批量输入、双输出 Tensor 和 Softmax Top-1。
- 两个 Adapter 都要求 Artifact 状态为 `engine_validated` 或 `deployed`。
- Engine 装载前使用 OpenSSL 重新计算 SHA256，并与 Model Registry 比较。
- 使用 MSVC 19.51、CUDA 13.3 和 TensorRT 10.16 完成编译、链接及缺失资产门禁测试。
- 已提供目标机 `trtexec` Engine 构建脚本、资产审计器和 0.5 个百分点绝对精度下降门禁。

## 当前真实环境与 Smoke 补测

- GPU：NVIDIA GeForce RTX 4080 Laptop GPU，Compute Capability 8.9。
- NVIDIA Driver 581.80；可用 TensorRT 10.16 和 11.1，项目构建基线为 10.16。
- 已收到 `DET-SMOKE-640` 和 `ATTR-SMOKE-224` 训练包；两者均只训练 3 epoch，
  且源模型卡明确标记 `release_eligible: false`。
- 两份 ONNX 已通过 checker 和 ONNX Runtime CPU 推理。
- 两份 FP16 Engine 已在本机使用 TensorRT 10.16.1.11 构建，并通过反序列化和推理。
- Attribute 在 45 张 test 裁剪上完成 PyTorch/ONNX/TensorRT 同集回归，
  四项 accuracy/F1 指标绝对下降均为 0。
- Detection 训练包未携带 75 张原始 test 图片，只完成一张确定性渲染图上的
  ONNX/TensorRT 数值一致性冒烟，无法重算 mAP/precision/recall。
- 详细证据位于
  `../../../training_artifacts/m2-final-20260728/validation/VALIDATION_SUMMARY.md`
  对应的工作区训练产物目录；这些 smoke 文件不复制为正式 v1 模型。

## 阻塞结论

Smoke ONNX、Engine、SHA256、模型卡和 Attribute 转换精度回归已经完成；
正式 v1 交付仍不能标记完成。当前阻塞是两个 Run 明确不具备发布资格、
Attribute 本身测试精度过低、Detection 原始封闭测试图片缺失，以及训练代码
Commit 证据缺失。

禁止采取以下替代方式：

- 用随机网络或空 ONNX 冒充训练模型；
- 为不存在的文件填写 SHA256；
- 将仅能装载 Engine 的冒烟测试写成精度通过；
- 使用不同数据切分生成不可比较的原始/ONNX/TensorRT指标。

## 解阻顺序

1. 交付 Detection 原始封闭 test 图片并按 manifest SHA256 校验。
2. 在正式冻结数据集上训练非 smoke 的 `vehicle-det-v1` 和 `vehicle-attr-v1`。
3. 记录可核验的 40 位训练代码 Commit，并提供完整模型卡和逐样本预测。
4. 在目标机重新构建正式 Engine。
5. 对同一封闭测试集生成原始、ONNX 和 TensorRT 报告。
6. 使用 `tools/compare_model_metrics.py` 验证绝对下降不超过 0.005。
7. 模型自身精度和转换门禁均通过后，将正式 Registry 升级为
   `engine_validated`；部署验收后再升级为 `deployed`。
