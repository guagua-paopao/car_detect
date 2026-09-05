# VCAS 云端训练工程

本目录实现 VCAS M2 末尾的训练链路验证，覆盖：

1. 按类别下载 Open Images V7 车辆子集；
2. 转换为 YOLO 检测数据并生成可追溯 Dataset Manifest；
3. 从车辆框生成二级属性裁剪标注模板；
4. 训练 YOLO11s 一级检测模型；
5. 训练 MobileNetV3-Large 车身类型/颜色双头模型；
6. 导出并验证 ONNX；
7. 生成训练日志、模型卡、指标和 SHA256。

AutoDL 从开机、上传、环境初始化、Smoke、正式四实验、断点续训到下载关机的完整步骤，
见 [AUTODL_RUNBOOK.md](AUTODL_RUNBOOK.md)。

正式训练入口：

```bash
bash training/scripts/bootstrap_autodl.sh
nohup bash training/scripts/run_formal_training.sh \
  > /root/autodl-tmp/vcas/manifests/formal-training.log 2>&1 &
```

四套 `training/configs/*.json` 是实际运行输入，不再只作说明。正式入口会执行
3,000 张 Detection、8,000 个 Attribute 裁剪的硬门禁，自动记录环境、完整命令、
代码/源码包版本、配置 SHA256、测试指标和 ONNX SHA256，并在中断后从 `last.pt`
自动恢复。

## 不变量

- 原始图片、裁剪、权重和运行产物不提交 Git。
- 检测、属性数据必须按 `camera_id/video_id/track_group` 分组切分。
- `poor` 裁剪不参与属性监督。
- `unknown` 是拒识结果；训练时屏蔽该属性头的损失，推理时由阈值产生。
- 云端只交付 ONNX；TensorRT Engine 在目标 Windows 主机生成。
- `DET-SMOKE-*`、`ATTR-SMOKE-*` 不是正式 `vehicle-*-v1` 发布物。

## 云端目录

```text
/root/autodl-tmp/vcas/
├── code/
├── datasets/
├── runs/
├── artifacts/
└── manifests/
```

## 1. 环境检查

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pip install -r training/requirements-cloud.txt
pip freeze > /root/autodl-tmp/vcas/manifests/requirements-lock.txt
```

不要在预装 PyTorch/CUDA 的云镜像中盲目升级 `torch`。

## 2. 下载检测 Pilot

执行下载前，必须已经完成 Open Images 用途和许可复核，并提供内部批准记录编号：

```bash
python training/scripts/download_openimages.py \
  --output-root /root/autodl-tmp/vcas/datasets/dataset-pilot-v1 \
  --labels config/vehicle_labels.v1.json \
  --train-samples 350 \
  --validation-samples 75 \
  --test-samples 75 \
  --seed 20260728 \
  --license-approval-ref VCAS-LIC-OI-001
```

## 3. 校验检测数据

```bash
python training/scripts/validate_training_inputs.py \
  --detection-root /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/detection \
  --min-detection-images 500 \
  --summary /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/detection-validation.json
```

## 4. 检测冒烟

```bash
python training/scripts/train_detector.py \
  --data /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/detection/vehicle_det_pilot.yaml \
  --model yolo11s.pt \
  --imgsz 640 \
  --epochs 3 \
  --batch 8 \
  --device 0 \
  --project /root/autodl-tmp/vcas/runs \
  --name DET-SMOKE-640
```

## 5. 生成属性裁剪模板

```bash
python training/scripts/prepare_attribute_crops.py \
  --detection-root /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/detection \
  --output-root /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/attributes \
  --max-crops 300 \
  --min-width 96
```

填写生成的 `attribute_manifest.csv`。未完成的空标签不能进入训练。

## 6. 属性冒烟

```bash
python training/scripts/validate_training_inputs.py \
  --attribute-csv /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/attributes/attribute_manifest.csv \
  --labels config/vehicle_labels.v1.json \
  --min-attribute-crops 300

python training/scripts/train_attribute.py \
  --manifest /root/autodl-tmp/vcas/datasets/dataset-pilot-v1/attributes/attribute_manifest.csv \
  --labels config/vehicle_labels.v1.json \
  --input-size 224 \
  --epochs 3 \
  --batch-size 32 \
  --device cuda \
  --output-dir /root/autodl-tmp/vcas/runs/ATTR-SMOKE-224

python training/scripts/export_attribute_onnx.py \
  --checkpoint /root/autodl-tmp/vcas/runs/ATTR-SMOKE-224/best.pt \
  --output /root/autodl-tmp/vcas/artifacts/vehicle-attr-smoke-224.onnx
```

## 7. 冒烟完成条件

- 检测和属性模型均完成至少 1 个 epoch；
- `best.pt`、`last.pt`、ONNX、指标和模型卡存在；
- ONNX Runtime 可加载并返回预期输出；
- 所有产物有 SHA256；
- 训练可以从 `last.pt` 恢复；
- 产物已下载或备份后，云实例已关机。

轻量属性冒烟的 300 个裁剪只验证工程链路，不替代 M1 原计划的
1,000 个属性裁剪正式门禁。
