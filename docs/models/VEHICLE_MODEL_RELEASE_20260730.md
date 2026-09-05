# 车辆模型 v1 正式交付记录

日期：2026-07-30

## 交付结论

- `vehicle-det-v1` 与 `vehicle-attr-v1` 已放入项目。
- ONNX 接口、类别顺序、SHA256、TensorRT Engine 和项目 C++ Adapter 均已验证。
- Model Registry 与运行时配置状态已升级为 `engine_validated`。
- Engine 仅适用于本机 Windows、RTX 4080 Laptop、TensorRT 10.16.1.11 环境。
- 尚未标记为 `deployed`；该状态需要真实摄像头或冻结视频回放验收。

## 数据与训练处理

- 人工复核优先样本 600 张：保留可识别样本 221 张，剔除不可识别样本 379 张。
- 从候选集人工复核并补充高质量样本 379 张，保持数据量不因剔除而下降。
- 属性模型最终使用 human-priority 数据版本 `dataset-final-v8`。
- 禁用 3,254 条不稳定的 VLM 车身类型监督；保留 810 条人工复核车身类型标签。
- 训练和导出预处理已与项目 C++ 运行时对齐：RGB、拉伸到 224×224、输入范围
  `[0,1]`，ImageNet mean/std 归一化嵌入属性 ONNX。
- 属性模型保留 `unknown` 拒识策略，不以低置信结果冒充准确分类。

## 冻结指标

检测模型：

- Precision：0.56565
- Recall：0.50245
- mAP50：0.56106
- mAP50-95：0.37975

属性模型独立测试集共 193 张：

- 车身类型 raw top-1 accuracy：31.09%
- 颜色 raw top-1 accuracy：65.28%
- 车身类型高置信 precision：100%（13/193 被选择，coverage 6.74%）
- 颜色高置信 precision：94.67%（75/193 被选择，coverage 38.86%）

项目发布目标按高置信 precision 门禁计算，不应把上述 raw top-1 指标写成 93%。
未达到置信门限的结果必须继续输出 `unknown`。

## 正式文件

| Artifact | 文件 | SHA256 |
| --- | --- | --- |
| Detection ONNX | `models/vehicle-det-v1.onnx` | `068ddae905311616cf1c5349a51fbcb70ab8dfffabf1894d59148a998fcd1fb1` |
| Attribute ONNX | `models/vehicle-attr-v1.onnx` | `a2576cafa90c7e4a0831ba1969d6710002f5c65e1b0413f5b67f98f2e0b82f00` |
| Detection Engine | `engines/vehicle-det-v1.engine` | `a35d41832e0503532ab4a994ff9613bf5475d88dcf6fbce9f83094cdddc61e12` |
| Attribute Engine | `engines/vehicle-attr-v1.engine` | `10dd7dd0107ede114c75af5996aba0ac7ea3dfbf374822df93902082dd63e435` |

## 接口冻结

- 检测输入：`images[1,3,960,960]`
- 检测输出：`output0[1,10,18900]`
- 检测类别：`car,bus,truck,motorcycle,vehicle,other`
- 属性输入：`images[batch,3,224,224]`，batch 为 1–16
- 属性输出：`body_type[batch,10]`、`color[batch,10]`
- 车身类型：
  `sedan,suv,mpv,van,pickup,bus,light_truck,heavy_truck,other,unknown`
- 颜色：
  `black,white,silver_gray,red,blue,green,yellow_orange,brown_beige,other,unknown`

## 已执行验证

1. ONNX checker 与本地接口检查通过。
2. 两份 FP16 Engine 构建、反序列化与真实 GPU 推理通过。
3. 项目 `TensorRtVehicleDetectionRunner` 和
   `TensorRtVehicleAttributeRunner` 使用真实 Engine 完成推理。
4. Model Registry validator、资产审计、标签契约和 C++ CTest 全部通过。

项目级真实 Engine 测试：

```powershell
$env:Path = "D:\TensorRT-10.16.1.11\bin;D:\TensorRT-10.16.1.11\lib;D:\GPU13.3\bin;" + $env:Path
F:\codex\project\out\build\backend-model-validation\vehicle_tensorrt_real_engine_test.exe F:\codex\project
```

## 剩余人工验收

只剩真实业务输入验收：提供一段冻结视频或可访问摄像头，在项目中回放并复核白天、
夜间、遮挡、小目标场景的检测召回率，以及属性 `unknown` coverage。通过后再把
Registry 状态从 `engine_validated` 升级为 `deployed`。
