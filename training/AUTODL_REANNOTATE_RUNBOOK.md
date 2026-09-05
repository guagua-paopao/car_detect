# VCAS 单数据集重标注与发布训练步骤

本流程只把本地仓库中的以下文件作为接口真源：

- `config/vehicle_labels.v1.json`
- `models/manifests/model_registry.v1.json`
- `include/server/vehicle_tensorrt_adapters.h`

云端旧代码和旧模型不得用于推断标签顺序。

## 固定发布接口

- Detection：YOLO11s、`1x3x960x960`、输入 `images`、物理输出
  `output0`，类别顺序为
  `car,bus,truck,motorcycle,vehicle,other`。
- Attribute：MobileNetV3-Large、`Nx3x224x224`、输入 `images`、输出
  `body_type,color`。
- Attribute 两个输出均为 10 维，顺序与
  `config/vehicle_labels.v1.json` 完全相同，包含末尾的 `unknown`。

## 1. 7B 教师模型校准

先在已经人工审核的 300 个裁剪上运行教师模型。禁止未经校准直接全量标注。

```bash
HF_ENDPOINT=https://hf-mirror.com \
/root/autodl-tmp/vcas/work/codex-annotation/venv-vlm/bin/python \
  /root/autodl-tmp/vcas/work/codex-annotation/evaluate_qwen_vl_local_contract.py \
  --manifest /root/autodl-tmp/vcas/datasets/dataset-v1/attributes/attribute_manifest_codex_annotated.csv \
  --output-jsonl /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-pilot/predictions.jsonl \
  --summary /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-pilot/summary.json \
  --cache-dir /root/autodl-tmp/vcas/work/codex-annotation/models/qwen25-vl-7b \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --limit 300 \
  --batch-size 2
```

只有 `body_type`、`color` 的高置信预测在人工对照集上达到 Precision
不低于 0.93 时，才允许进入自动接收阶段。

## 2. 全量属性预标注

校准通过后，去掉 `--limit` 运行同一命令并使用新的全量输出目录。JSONL
逐行落盘并支持断点续跑。

## 3. 校准筛选和人工复核队列

```bash
cd /root/autodl-tmp/vcas/code-20260730-r6

/root/miniconda3/bin/python training/scripts/merge_vlm_annotations.py \
  --manifest /root/autodl-tmp/vcas/datasets/dataset-v1/attributes/attribute_manifest_codex_annotated.csv \
  --predictions \
    /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-approved-pilot/predictions.jsonl \
    /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-pending-train/predictions.jsonl \
  --secondary-predictions-csv /root/autodl-tmp/vcas/work/codex-annotation/prelabels_mobilenet_full.csv \
  --output /root/autodl-tmp/vcas/datasets/dataset-release-v1/attributes/attribute_manifest.csv \
  --review-queue /root/autodl-tmp/vcas/datasets/dataset-release-v1/attributes/manual_review_queue.csv \
  --report /root/autodl-tmp/vcas/datasets/dataset-release-v1/attributes/merge_report.json \
  --labels config/vehicle_labels.v1.json \
  --target-precision 0.93 \
  --materialize-crops
```

自动流程始终保留原人工审核行，不覆盖 Validation/Test。以下内容必须人工完成：

1. `manual_review_queue.csv` 中的 Validation/Test 全部复核；
2. 自动教师未达到 0.93 校准精度的类别全部复核；
3. Detection 的 `vehicle`、`other` 必须补足真实代表样本，禁止用空类别冒充支持；
4. 随机抽查至少 10% 自动接收样本。

教师与旧审核标签的冲突可先生成图集：

```bash
/root/miniconda3/bin/python training/scripts/build_annotation_review_sheets.py \
  --manifest /root/autodl-tmp/vcas/datasets/dataset-v1/attributes/attribute_manifest_codex_annotated.csv \
  --predictions /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-pilot/predictions.jsonl \
  --output-dir /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-pilot/review \
  --only-approved
```

人工在 `review_queue.csv` 的四个 `reviewed_*` 列填入最终值后应用：

```bash
/root/miniconda3/bin/python training/scripts/apply_annotation_reviews.py \
  --manifest /root/autodl-tmp/vcas/datasets/dataset-v1/attributes/attribute_manifest_codex_annotated.csv \
  --reviews /root/autodl-tmp/vcas/work/codex-annotation/evaluations/qwen25-vl-7b-pilot/review/review_queue.csv \
  --output /root/autodl-tmp/vcas/datasets/dataset-v1/attributes/attribute_manifest_rereviewed.csv \
  --labels config/vehicle_labels.v1.json
```

## 4. Detection 训练集漏框审计

此步骤创建新数据集，旧数据集不会被覆盖。原 Open Images 标注优先，只为
Train 增加 YOLO11x 高置信且无类别冲突的漏框。

```bash
/root/miniconda3/bin/python training/scripts/audit_detection_labels.py \
  --source-root /root/autodl-tmp/vcas/datasets/dataset-v1/detection \
  --output-root /root/autodl-tmp/vcas/datasets/dataset-release-v1/detection \
  --labels config/vehicle_labels.v1.json \
  --teacher yolo11x.pt \
  --imgsz 1280 \
  --confidence 0.70 \
  --device 0 \
  --report /root/autodl-tmp/vcas/datasets/dataset-release-v1/detection/audit_report.json
```

## 5. 数据门禁

```bash
/root/miniconda3/bin/python training/scripts/validate_training_inputs.py \
  --detection-root /root/autodl-tmp/vcas/datasets/dataset-release-v1/detection \
  --attribute-csv /root/autodl-tmp/vcas/datasets/dataset-release-v1/attributes/attribute_manifest.csv \
  --labels config/vehicle_labels.v1.json \
  --min-detection-images 5000 \
  --min-attribute-crops 8000 \
  --require-approved \
  --summary /root/autodl-tmp/vcas/manifests/training-input-validation.release.json
```

门禁失败时不得启动正式训练。

## 6. 单组正式训练

```bash
cd /root/autodl-tmp/vcas/code-20260730-r6

VCAS_ROOT=/root/autodl-tmp/vcas \
CODE_ROOT=/root/autodl-tmp/vcas/code-20260730-r6 \
DATASET_ROOT=/root/autodl-tmp/vcas/datasets/dataset-release-v1 \
ATTRIBUTE_MANIFEST=/root/autodl-tmp/vcas/datasets/dataset-release-v1/attributes/attribute_manifest.csv \
SOURCE_REVISION_FILE=/root/autodl-tmp/vcas/manifests/SOURCE_REVISION_R6 \
nohup bash training/scripts/run_release_training.sh \
  > /root/autodl-tmp/vcas/manifests/release-training.log 2>&1 &
```

流程只训练一组最终模型：YOLO11s/960 和 MobileNetV3-Large/224。

## 7. 最终检查和下载

正式产物：

```text
/root/autodl-tmp/vcas/release/artifacts/vehicle-det-v1.onnx
/root/autodl-tmp/vcas/release/artifacts/vehicle-attr-v1.onnx
/root/autodl-tmp/vcas/release/manifests/release-onnx-contract.json
```

`release-onnx-contract.json` 必须为 `status=pass`，并确认：

- Detection 输出含 `4 + 6 = 10` 个特征；
- Attribute 输出名为 `body_type`、`color`，形状均为 `[batch,10]`；
- 输入尺寸、标签顺序、SHA256 均与本地项目契约一致。

下载并复核 SHA256 后，才可停止或释放云实例。
