# VCAS 模型交付回补状态

核验日期：2026-07-28

## 已完成

- `TensorRtVehicleDetectionRunner` 已实现真实 TensorRT Engine 反序列化、CUDA 输入输出、YOLO 原始输出解码和 NMS。
- `TensorRtVehicleAttributeRunner` 已实现动态批量输入、双输出 Tensor 和 Softmax Top-1。
- 两个 Adapter 都要求 Artifact 状态为 `engine_validated` 或 `deployed`。
- Engine 装载前使用 OpenSSL 重新计算 SHA256，并与 Model Registry 比较。
- 使用 MSVC 19.51、CUDA 13.3 和 TensorRT 10.16 完成编译、链接及缺失资产门禁测试。
- 已提供目标机 `trtexec` Engine 构建脚本、资产审计器和 0.5 个百分点绝对精度下降门禁。

## 当前真实环境

- GPU：NVIDIA GeForce RTX 4080 Laptop GPU，Compute Capability 8.9。
- CUDA：13.3。
- 可用 TensorRT：10.16 和 11.1；项目当前构建基线为 10.16。
- Detection ONNX：缺失。
- Attribute ONNX：缺失。
- Detection/Attribute Engine：缺失。
- 真实数据版本、训练 Run、指标和逐样本预测：缺失。

## 阻塞结论

真实 ONNX、Engine、SHA256、模型卡和精度回归不能标记完成。原因不是 Adapter 或目标机工具缺失，而是 M1 数据试标、正式授权数据、模型训练和导出尚未完成。

禁止采取以下替代方式：

- 用随机网络或空 ONNX 冒充训练模型；
- 为不存在的文件填写 SHA256；
- 将仅能装载 Engine 的冒烟测试写成精度通过；
- 使用不同数据切分生成不可比较的原始/ONNX/TensorRT指标。

## 解阻顺序

1. 完成授权数据试标并冻结 `dataset-v1`。
2. 训练 `vehicle-det-v1` 和 `vehicle-attr-v1`。
3. 导出真实 ONNX，并写入 SHA256、训练 Run、数据版本和代码 Commit。
4. 填写模型卡与原始模型指标。
5. 在目标机运行 `scripts/build_vehicle_engines.ps1`。
6. 对同一封闭测试集生成原始、ONNX 和 TensorRT 报告。
7. 使用 `tools/compare_model_metrics.py` 验证绝对下降不超过 0.005。
8. 通过后将 Registry 状态升级为 `engine_validated`；部署验收后再升级为 `deployed`。
