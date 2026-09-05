# VCAS 数据来源与许可台账

最后核验日期：2026-07-28

本台账不是法律意见。任何候选来源在进入训练、验证或测试 Manifest 前，必须将状态改为已批准，并保存授权、许可文本、访问日期、用途限制和署名要求。

| 来源 ID | 来源 | 官方许可信息 | M1 状态 | 处理要求 |
|---|---|---|---|---|
| TARGET-CAMERA | 授权现场摄像头 | 由数据提供方书面授权决定 | pending_authorization | 未保存授权编号前禁止进入数据集；执行最小化和留存策略。 |
| OPEN-IMAGES-V7 | Open Images V7 | 标注 CC BY 4.0；图像列为 CC BY 2.0，官方提示逐图核验许可 | review_required | 记录每张图的原始 ID、许可、作者/来源和下载日期；禁止把数据集收录本身当作图像许可保证。 |
| MIO-TCD | MIO-TCD | CC BY-NC-SA 4.0 | review_required_noncommercial | 仅适用于当前非商业学习项目；记录署名、非商业和相同方式共享义务。 |
| COMPCARS | CompCars | 仅非商业研究，限制复制、发布和再分发 | excluded_from_mvp | MVP 不依赖；如二期研究使用，需单独批准和隔离存储。 |
| PKU-VD | PKU VehicleID / PKU-VD | 仅学术用途、需要签署协议、禁止商业使用 | excluded_from_mvp | 当前个人学习仓库不默认满足申请条件；不得下载或使用未获批准的数据。 |

## 官方来源

- Open Images V7：https://storage.googleapis.com/openimages/web/index.html
- Open Images 许可说明：https://github.com/openimages/dataset/blob/main/READMEV3.md
- MIO-TCD：https://tcd.miovision.com/challenge/dataset.html
- CompCars：https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- PKU-VD：https://pkuml.org/resources/pku-vds.html

## 批准记录最小字段

每个实际使用的来源至少保存：

- `source_id`
- 数据所有者或发布方
- 官方主页和许可文本 URL
- 许可或授权编号
- 访问和复核日期
- 允许用途、禁止用途和到期条件
- 署名与 ShareAlike 要求
- 是否允许再分发原始数据和派生数据
- 数据保存位置、负责人和删除方式

Manifest 校验器拒绝 `review_required` 或 `blocked` 来源的样本，以防止“先训练、后补许可”。
