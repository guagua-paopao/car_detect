# VCAS M1 数据标注规范

版本：`vehicle-labels-v1`

## 1. 总体原则

- 只处理来源和用途已经批准的数据。
- 同一摄像头、视频段或车辆轨迹不得跨训练、验证和测试集合。
- 看不清时标记 `unknown`，禁止凭主观猜测补全类型或颜色。
- 原始图像和裁剪不进入 Git；Git 只保存标签定义、Manifest、统计和验证证据。
- 不采集车辆所有人信息；车牌不是本项目标签，包含车牌的证据图按最小化策略处理。

## 2. 一级检测标注

每个检测样本至少包含：

- `bbox_xyxy_norm`：归一化 `[x1, y1, x2, y2]`，满足 `0 <= x1 < x2 <= 1` 和 `0 <= y1 < y2 <= 1`
- `vehicle_class`：`car`、`bus`、`truck`、`motorcycle`、`vehicle` 或 `other`
- `occluded`：车辆主体是否被明显遮挡
- `truncated`：车辆框是否超出图像边界
- `camera_id`、`video_id`、`track_group`
- 样本文件 SHA256 和来源编号

边框应覆盖可见车辆主体，不额外包含大面积道路、行人或相邻车辆。无法稳定细分时使用 `vehicle` 或 `other`，一级检测不使用 `unknown`。

## 3. 二级属性标注

属性样本必须关联车辆轨迹，并包含：

- `body_type`
- `color`
- `crop_quality`
- `viewpoint`
- `blur`、`occluded`、`truncated`、`night`

### 3.1 车身类型

允许值：

`sedan`、`suv`、`mpv`、`van`、`pickup`、`bus`、`light_truck`、`heavy_truck`、`other`、`unknown`

- 仅凭颜色、徽标或车牌不得推断车身类型。
- 车辆过小、主体不可辨、严重遮挡或视角不足时使用 `unknown`。
- 品牌、车系和年款不进入当前标签。

### 3.2 颜色

允许值：

`black`、`white`、`silver_gray`、`red`、`blue`、`green`、`yellow_orange`、`brown_beige`、`other`、`unknown`

- `gray`、`grey` 和 `silver` 统一归一为 `silver_gray`。
- 夜间偏色、过曝、强反光或仅可见局部时使用 `unknown`。
- 不使用会改变真实颜色语义的增强结果作为人工标注依据。

### 3.3 裁剪质量

- `good`：主体完整、尺寸足够、清晰且属性可判断。
- `usable`：存在轻度模糊、遮挡、截断或曝光问题，但仍可可靠判断至少一个属性。
- `poor`：属性不可可靠判断。`body_type` 和 `color` 必须同时为 `unknown`。

### 3.4 视角

允许值：

`front`、`rear`、`side`、`front_three_quarter`、`rear_three_quarter`、`unknown`

## 4. 分组与切分

Manifest 固定使用以下三个分组键：

1. `camera_id`
2. `video_id`
3. `track_group`

任一键值一旦进入某个 Split，该键值关联的全部样本只能存在于同一 Split。默认比例为训练 70%、验证 15%、测试 15%，但比例可以调整，分组原则不能取消。

## 5. M1 试标

- 检测帧：500 张
- 属性裁剪：1,000 个
- 白天、夜间、遮挡、模糊、小目标和困难负样本均需覆盖
- 至少抽取 10% 样本进行独立复核
- 所有冲突必须记录最终裁决及原因

试标退出条件：

- Manifest 校验器无错误
- 不存在分组泄漏
- 标签值全部来自 `config/vehicle_labels.v1.json`
- 来源、授权、许可和 SHA256 完整
- 主要歧义形成示例并补充到本规范
