# VCAS AutoDL 训练操作手册

适用实例：AutoDL `b197439d52-e206ffbd`，内蒙 B 区，单卡 `vGPU 32GB`。

这台实例足以执行 `DET-A-640`、`ATTR-A-224`，也可尝试
`DET-B-960`、`ATTR-B-256`。四个实验采用串行执行，避免单卡显存争用。

## 一、开始前的真实状态

当前工程已经具备训练、验证、ONNX 导出和 TensorRT 交付链路，但现有数据是
Pilot：

- Detection：500 张，1,241 个框；
- Attribute：300 个裁剪，其中可监督质量样本 201 个；
- 当前 Smoke 权重只训练了 3 epoch，不能作为正式模型。

因此分两条路径：

1. **立即可做：** 在 AutoDL 复现 Smoke，确认云环境、上传、断点续训和下载链路；
2. **正式训练：** 补齐并冻结至少 3,000 张目标域检测图、8,000 个属性裁剪，
   再运行四套正式实验。

脚本会硬性执行该数据门禁，不会把 Pilot 误标为正式模型。
Attribute 门禁只统计 `review_status=approved`、非 `poor` 且至少有一个有效属性标签的
可监督裁剪；预标注但仍为 `pending` 的数据不会被算入正式训练规模。

## 二、需要人工完成的步骤

### 1. 开机

在 AutoDL 控制台点击实例 `b197439d52-e206ffbd` 的“开机”。这是按量计费动作，
必须由账号持有人确认。

开机后记录：

- SSH 主机、端口和用户名；
- JupyterLab 入口；
- 预计自动关机时间。

不要把密码、验证码或私钥写进项目文件。

### 2. 上传代码包

将 `vcas-autodl-code-20260730.zip` 上传到 `/root/autodl-tmp/`，然后在
AutoDL 终端执行：

```bash
mkdir -p /root/autodl-tmp/vcas/code
unzip -q /root/autodl-tmp/vcas-autodl-code-20260730.zip \
  -d /root/autodl-tmp/vcas/code
cd /root/autodl-tmp/vcas/code
```

也可以使用本机 `scp`，但 SSH 主机和端口必须以控制台显示为准。

### 3. 上传正式数据

正式数据必须放成以下结构：

```text
/root/autodl-tmp/vcas/datasets/dataset-v1/
├── detection/
│   ├── images/{train,validation,test}/
│   ├── labels/{train,validation,test}/
│   └── vehicle_det_v1.yaml
└── attributes/
    ├── crops/...
    └── attribute_manifest.csv
```

该步骤涉及授权视频、图片和人工标注，只能由数据负责人完成。禁止上传未获授权的
人脸、车牌或商业受限数据。切分必须按 `camera_id/video_id/track_group` 分组，
同一轨迹不得跨 train/validation/test。

## 三、AutoDL 环境初始化

建议选择 AutoDL 已匹配 CUDA 的 PyTorch 镜像。不要另行升级 `torch` 和
`torchvision`。

```bash
cd /root/autodl-tmp/vcas/code
bash training/scripts/bootstrap_autodl.sh \
  2>&1 | tee /root/autodl-tmp/vcas/manifests/bootstrap.log
```

脚本会：

- 安装工程依赖并执行 `pip check`；
- 校验 CUDA、显存和可用磁盘；
- 保存 GPU、驱动、Python、PyTorch、CUDA、cuDNN 证据；
- 生成依赖锁和源码文件 SHA256；
- 生成正式 Run 使用的 `SOURCE_REVISION`。

通过标准是 `cloud-environment.json` 中 `status=pass`。

## 四、先跑 Smoke

如果正式数据还未冻结，先准备 500 张 Open Images Pilot。下载前必须确认研究用途
和许可记录编号：

```bash
cd /root/autodl-tmp/vcas/code
python training/scripts/download_openimages.py \
  --output-root /root/autodl-tmp/vcas/datasets/dataset-pilot-v1 \
  --labels config/vehicle_labels.v1.json \
  --train-samples 350 \
  --validation-samples 75 \
  --test-samples 75 \
  --seed 20260728 \
  --license-approval-ref VCAS-LIC-OI-001
```

属性裁剪需要人工标注。若已有已批准的 `attribute_manifest.csv` 和裁剪，应整体上传到
Pilot 的 `attributes/`。然后执行：

```bash
RUN_ATTRIBUTE=1 bash training/scripts/smoke_cloud.sh \
  2>&1 | tee /root/autodl-tmp/vcas/manifests/smoke.log
```

如果还没有属性标注，先使用 `RUN_ATTRIBUTE=0` 只验证 Detection。

## 五、正式四实验训练

如果目标是只训练并发布一个最终组合，优先使用单方案入口：

```bash
CODE_ROOT=/root/autodl-tmp/vcas/code-20260730-r4 \
DATASET_ROOT=/root/autodl-tmp/vcas/datasets/dataset-release-v1 \
ATTRIBUTE_MANIFEST=/root/autodl-tmp/vcas/datasets/dataset-release-v1/attributes/attribute_manifest.csv \
SOURCE_REVISION_FILE=/root/autodl-tmp/vcas/manifests/SOURCE_REVISION.r4 \
  nohup bash training/scripts/run_release_training.sh \
  > /root/autodl-tmp/vcas/manifests/release-training.log 2>&1 &
```

该入口只训练 `DET-RELEASE-960` 和 `ATTR-RELEASE-256`，两者使用同一个
`dataset-release-v1`，最终形成一个级联发布包。数据门禁未通过时不会启动 GPU 训练。

数据负责人确认 `dataset-v1` 冻结后执行：

```bash
cd /root/autodl-tmp/vcas/code
nohup bash training/scripts/run_formal_training.sh \
  > /root/autodl-tmp/vcas/manifests/formal-training.log 2>&1 &
echo $! > /root/autodl-tmp/vcas/manifests/formal-training.pid
```

如果审核完成的正式清单使用了其他文件名，通过环境变量显式指定：

```bash
ATTRIBUTE_MANIFEST=/root/autodl-tmp/vcas/datasets/dataset-v1/attributes/attribute_manifest_reviewed.csv \
  nohup bash training/scripts/run_formal_training.sh \
  > /root/autodl-tmp/vcas/manifests/formal-training.log 2>&1 &
```

查看状态：

```bash
tail -f /root/autodl-tmp/vcas/manifests/formal-training.log
nvidia-smi
```

脚本顺序运行：

1. `DET-A-640`：YOLO11s，640，100 epoch；
2. `DET-B-960`：YOLO11s，960，100 epoch；
3. `ATTR-A-224`：MobileNetV3-Large 双头，224，40 epoch；
4. `ATTR-B-256`：MobileNetV3-Large 双头，256，40 epoch。

每套实验都从 `training/configs/*.json` 读取参数，记录完整命令、源码版本、数据版本、
配置 SHA256 和结果。再次执行同一命令时会跳过已完成实验，存在 `last.pt` 的未完成实验
会自动恢复。

### OOM 处理

先保留输入分辨率和数据版本，只降低 Batch：

- Detection：把配置中的 `batch` 从 `-1` 改为 `8`，仍 OOM 再改 `4`；
- Attribute：从 `64` 改为 `32`，仍 OOM 再改 `16`。

每次改动必须复制配置为新实验 ID，不得覆盖原实验记录。

## 六、完成、下载和关机

成功后生成：

```text
/root/autodl-tmp/vcas/
├── runs/                         # best.pt、last.pt、日志和测试指标
├── artifacts/                    # Detection/Attribute ONNX
├── manifests/                    # 环境、配置、命令和 SHA256
└── vcas-formal-training-results.tar.gz
```

先校验：

```bash
cd /root/autodl-tmp/vcas
sha256sum vcas-formal-training-results.tar.gz
ls -lh vcas-formal-training-results.tar.gz*
```

把 `.tar.gz` 和旁边的 `.json` 下载到本地，核对 SHA256 后再关机。关机前检查
`formal-training.log` 最后一行含 `PASS: all formal VCAS experiments completed`。

关闭实例不会立即清空数据，但控制台提示连续关机 15 天会释放实例，释放后数据不可恢复。
因此正式产物必须先完成本地或对象存储备份。

## 七、模型选择和发布门槛

- Detection：先比较目标域白天/夜间 Recall，再比较 mAP50:95 和目标机延迟；
- Attribute：类型和颜色 Macro-F1 均需达标，不能用平均分掩盖单项失败；
- 如果关键指标差异小于 1 个百分点，选择 640/224 的低延迟组合；
- 阈值只在 validation 调整，test 只运行最终候选；
- ONNX 与 TensorRT 相对原始模型的同集指标绝对下降不得超过 0.005；
- 初始属性拒识阈值：类型 0.75、颜色 0.70，目标是高置信 Precision ≥ 93%；
- Smoke、缺少封闭 test、缺少数据/源码 SHA256 的结果一律不得发布。
