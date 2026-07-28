# VCAS M1 数据规范

## 状态

- 阶段：M1
- 状态：in_progress
- 开始日期：2026-07-28
- 基线分支：`agent/vcas-m0-foundation`
- 基线提交：`36f1007e09fe46b761392fff394e222192099545`

## 目标

1. 冻结 `vehicle-labels-v1`。
2. 建立数据来源和许可门禁。
3. 建立 Dataset Manifest v1。
4. 自动拒绝跨摄像头、视频或轨迹分组的数据泄漏。
5. 为 500 张检测帧和 1,000 个属性裁剪试标建立可执行规范。

## 新增接口

- `config/vehicle_labels.v1.json`
- `api/schemas/dataset_manifest.v1.schema.json`
- `data/manifests/dataset_manifest.v1.example.json`
- `tools/validate_dataset_manifest.py`

## 许可结论

- Open Images V7 和 MIO-TCD 仅作为候选，完成许可复核前不得进入实际 Manifest。
- CompCars 和 PKU-VD 不纳入 MVP。
- 现场摄像头数据必须先记录书面授权编号。

## M1 退出条件

- [x] 标签、别名和 unknown 策略进入版本化配置
- [x] Dataset Manifest Schema、示例和校验器完成
- [x] 泄漏、非法路径、非法标签和未批准来源负向测试完成
- [x] 标注规范、许可台账和试标计划完成
- [ ] 完成 500 张检测帧和 1,000 个属性裁剪试标
- [ ] 形成 `dataset-pilot-v1` 并通过独立复核
- [ ] 批准正式 `dataset-v1` 的来源和许可

## 回滚

本阶段只新增数据契约、验证工具和文档，不包含原始数据。回滚不会影响 M0 运行时接口。
