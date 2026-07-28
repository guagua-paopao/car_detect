# VCAS M1 试标计划

## 目标

用小规模试标验证标签是否可执行、Manifest 是否可追溯，以及按摄像头、视频和轨迹分组的切分流程是否能阻止数据泄漏。

## 输入

- 已批准的数据来源
- 冻结的 `vehicle-labels-v1`
- 500 张检测帧
- 1,000 个属性裁剪

## 工作步骤

1. 建立来源与许可记录。
2. 为原始视频和导出样本计算 SHA256。
3. 分配全局唯一的 `camera_id`、`video_id` 和 `track_group`。
4. 按标注规范完成检测框与属性标注。
5. 独立复核至少 10% 样本。
6. 先按分组分配 Split，再生成样本级 Manifest。
7. 运行 `tools/validate_dataset_manifest.py`。
8. 汇总类别分布、unknown 比例、质量分布和冲突案例。

## 退出成果

- `dataset-pilot-v1` Manifest
- 来源与许可批准记录
- 标签分布与质量统计
- 标注冲突和裁决记录
- 分组泄漏校验通过记录
- 是否进入正式 `dataset-v1` 的评审结论
