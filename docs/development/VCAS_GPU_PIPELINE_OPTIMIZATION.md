# VCAS GPU 推理链路优化报告

## 1. 范围与结论

本轮针对生产车辆检测器的完整链路进行优化：测试视频实时推流、RTSP 拉流、
FFmpeg 解码、预处理、TensorRT 推理、后处理和精简结果输出。输入视频、模型、
阈值和输出语义保持不变。

最终 RTSP 三轮基准的中位数结果：模型处理均值由 `22.663 ms` 降至
`8.678 ms`（降低 `61.71%`），P95 由 `26.834 ms` 降至 `10.621 ms`
（降低 `60.42%`）。端到端吞吐约 `24.85 FPS`，由 25 FPS 实时输入封顶。

## 2. 固定测试条件

- GPU：NVIDIA GeForce RTX 4080 Laptop GPU，12,282 MiB
- 驱动：581.80
- CUDA Toolkit：13.3
- TensorRT：10.16.1.11
- OpenCV：4.12.0
- 模型：`engines/vehicle-det-v1.engine`
- 输入尺寸：模型 `1x3x960x960`；视频 `1280x720`
- 视频：`demo/rtsp_standard/output/vcas_rtsp_demo_60s.mp4`
- 视频 SHA256：`27DFAC5D82A857852B9533B3821A96777D2EC5F370C6740D005359CD8E1E5086`
- 编码：H.264 High / YUV420P / 25 FPS / 60 秒
- 阈值：confidence `0.25`，class-aware NMS IoU `0.45`
- 正式基准：每次先预热 200 帧，再测 2,000 帧，运行 3 次并取各轮统计量的中位数

本机已有的服务/worker 进程在基线和优化测试期间均未停止，因此 RTSP 指标反映
共享 GPU 下的真实运行状态。独立文件基准用于隔离和归因各组 CUDA 修改。

## 3. RTSP 模拟与复现

本地 MediaMTX 使用项目内置压缩包启动。FFmpeg 以 `-re -stream_loop -1`、
视频流复制和 TCP RTSP 按原始 25 FPS 循环推送，不重新编码。

```powershell
cd F:\codex\project
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_local_rtsp.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_vehicle_rtsp_benchmark.ps1 `
  -BuildDir .\out\build\gpu-pipeline-validation `
  -EvidenceDir .\reports\gpu-pipeline-optimization\rtsp-final
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_local_rtsp.ps1
```

30 分钟验收命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_vehicle_rtsp_benchmark.ps1 `
  -BuildDir .\out\build\gpu-pipeline-validation `
  -Iterations 45000 -WarmupFrames 200 -Runs 1 `
  -EvidenceDir .\reports\gpu-pipeline-optimization\rtsp-soak-30m
```

## 4. 实现改动

### GPU 预处理

- 将 resize/letterbox、BGR/RGB 转换、`uint8 -> float`、`/255` 归一化和
  HWC->RGB NCHW 排列融合为一个 CUDA kernel。
- 原始 BGR 数据先复制到可复用的 pinned host staging buffer，再使用同一
  CUDA stream 异步 H2D；每帧 H2D 从 10,800 KiB 浮点张量降为 2,700 KiB
  原始字节。
- TensorRT 输入、输出、pinned host 输出和 CUDA 事件按容量复用；稳态不再
  每帧 `cudaMalloc/cudaFree`。

### GPU 后处理

- 在 GPU 完成输出解码、置信度过滤、类别选择、坐标还原、CUB 分数排序和
  class-aware greedy NMS。
- CPU 只接收最终检测数和紧凑检测结构；代表帧 D2H 从 756,000 字节降至
  196 字节（检测数量不同时紧凑结果字节数会变化）。
- 保留一次为获取可变长紧凑结果所需的同步，不在循环内执行全量输出
  `.cpu()`/`.numpy()` 等隐式搬运。

### 兼容性

- `TensorRtDetectionOptions` 新增默认开启的 `use_gpu_preprocess` 和
  `use_gpu_postprocess`；两者关闭时继续走原 CPU 预处理/后处理路径。
- 公共检测请求、返回结构、配置格式、模型权重、分辨率、阈值及功能不变。

## 5. 可归因检查点（独立文件源）

| 检查点 | 均值 ms | P95 ms | 可持续 FPS | 预处理 ms | 后处理 ms | H2D 字节 | D2H 字节 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU 基线 | 11.401 | 13.284 | 85.17 | 5.106 | 0.188 | 11,059,200 | 756,000 |
| 仅 GPU 预处理 | 3.019 | 3.567 | 331.06 | 0.202（kernel 0.099） | 0.147 CPU | 2,764,800 | 756,000 |
| GPU 预处理 + 后处理 | 2.791 | 3.040 | 358.00 | 0.190（kernel 0.090） | 0.112（kernel 0.110） | 2,764,800 | 196 |

一次额外实验先在 GPU 按阈值压缩候选，再只排序有效候选。它需要在排序前读取
动态候选数并同步 CUDA stream，导致均值退化到 `3.734 ms`、FPS 降至
`267.64`，因此已回退，没有进入最终实现。

证据：

- `reports/gpu-pipeline-optimization/baseline-isolated.json`
- `reports/gpu-pipeline-optimization/gpu-preprocess-isolated.json`
- `reports/gpu-pipeline-optimization/gpu-prepost-isolated.json`
- `reports/gpu-pipeline-optimization/gpu-prepost-compact-isolated.json`

## 6. 完整 RTSP 基准

| 指标（三轮中位数） | 基线 | 优化后 | 变化 |
|---|---:|---:|---:|
| 处理均值 | 22.663 ms | 8.678 ms | -61.71% |
| 处理 P95 | 26.834 ms | 10.621 ms | -60.42% |
| 实时吞吐 | 24.908 FPS | 24.850 FPS | 输入 25 FPS 封顶 |
| 预处理均值 | 5.120 ms CPU | 0.478 ms（GPU kernel 0.331） | -90.66% |
| H2D 均值 | 3.799 ms | 0.371 ms | -90.24% |
| 推理均值 | 10.381 ms | 6.896 ms | -33.57%（共享 GPU 波动） |
| 后处理均值 | 0.217 ms CPU | 0.356 ms（GPU kernel 0.353） | 低频调用时 kernel 启动占优 |
| D2H 均值 | 0.439 ms | 0.172 ms | -60.95% |

RTSP 拉流和解码在线程中异步进行。优化后每次等待下一帧的均值约 `31.53 ms`，
加上 `8.68 ms` 模型处理后对应约 25 FPS 的源节拍。采集状态为 `running`，
`open_count=1`，`reconnect_count=0`。

进程遥测的均值由 CPU `4.104%` 降至 `3.095%`；GPU 利用率由 `35.348%`
变为 `36.910%`。`nvidia-smi` 的显存指标包含测试前已存在的 server、worker
和桌面进程，因此以相同环境对比，并在长稳测试中同时记录目标进程 working set
的首尾窗口变化。

| 资源指标 | 基线均值 / 峰值 | 优化后均值 / 峰值 |
|---|---:|---:|
| 目标进程 CPU | 4.104% / 9.363% | 3.095% / 10.626% |
| 目标进程 working set | 344.939 / 353.953 MiB | 344.489 / 348.391 MiB |
| 全局 GPU 利用率 | 35.348% / 52% | 36.910% / 46% |
| 全局 GPU 显存 | 2,982.888 / 2,998 MiB | 2,976.871 / 3,162 MiB |

证据：

- `reports/gpu-pipeline-optimization/rtsp-baseline/benchmark.json`
- `reports/gpu-pipeline-optimization/rtsp-baseline/telemetry-summary.json`
- `reports/gpu-pipeline-optimization/rtsp-final/benchmark.json`
- `reports/gpu-pipeline-optimization/rtsp-final/telemetry-summary.json`

## 7. 硬件解码评估

本机 FFmpeg 支持 `cuda` 和 `h264_cuvid`。对相同 RTSP 流解码 500 帧，并仍
输出生产接口需要的 CPU YUV420P：

| 路径 | FFmpeg user+system CPU | 每帧 CPU | wall | max RSS |
|---|---:|---:|---:|---:|
| 软件 H.264 | 1.281 s | 2.562 ms | 20.056 s | 118.9 MiB |
| CUVID + hwdownload | 1.016 s | 2.032 ms | 19.871 s | 390.4 MiB |

CUDA 解码减少约 20.7% FFmpeg CPU 时间，但实时 wall 延迟没有实质改善，进程
内存增加约 271.5 MiB。当前生产接口把 FFmpeg YUV420P pipe 转为 CPU BGR
`cv::Mat`，所以 CUVID 仍必须 `hwdownload`，不是低拷贝路径；解码也不是当前
瓶颈。因此本轮不默认开启硬解。后续若把 `FrameEnvelope` 扩展为 CUDA/NVDEC
surface，再直接交给预处理 kernel，才值得重新评估。

证据：

- `reports/gpu-pipeline-optimization/decode-software.log`
- `reports/gpu-pipeline-optimization/decode-cuda.log`

## 8. 正确性与回归

- 固定视频抽取 50 帧、步长 20，逐帧比较 CPU fallback 与 GPU 路径。
- 共匹配 409 个检测；类别和检测数一致，最大置信度差 `0`，最大坐标差
  `0.949 px`，最小 IoU `0.9575`（极窄小框对亚像素差更敏感）。
- 新构建目录的 28 个 CTest 全部通过；11 项外部 PostgreSQL/Redis/集成环境
  测试按项目既有规则报告为 skipped。
- `scripts/test_all.ps1` 的 CTest、Web Admin 和 8 个车辆 Python 合约套件全部通过。

复现：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_backend.ps1 `
  -BuildDir .\out\build\gpu-pipeline-validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_all.ps1 `
  -BuildDir .\out\build\gpu-pipeline-validation
$env:Path = "D:\GPU13.3\bin;D:\TensorRT-10.16.1.11\lib;D:\libs\opencv\build\x64\vc16\bin;$env:Path"
.\out\build\gpu-pipeline-validation\vehicle_gpu_pipeline_parity_test.exe `
  (Get-Location).Path .\demo\rtsp_standard\output\vcas_rtsp_demo_60s.mp4
```

## 9. 交付文件

| 文件 | 用途 |
|---|---|
| `include/server/vehicle_tensorrt_adapters.h` | 阶段计时、GPU 路径开关和兼容 CPU fallback |
| `src/server/vehicle_tensorrt_adapters.cpp` | 持久缓冲区、异步执行及 GPU 预/后处理接入 |
| `include/server/vehicle_cuda_preprocess.h` | CUDA 预处理接口 |
| `src/server/vehicle_cuda_preprocess.cu` | 融合 letterbox/颜色/归一化/NCHW kernel |
| `include/server/vehicle_cuda_postprocess.h` | CUDA 后处理接口 |
| `src/server/vehicle_cuda_postprocess.cu` | decode/filter/sort/NMS/紧凑回传 |
| `tests/vehicle_tensorrt_benchmark.cpp` | 文件与生产 FFmpeg RTSP 全链路阶段基准 |
| `tests/vehicle_gpu_pipeline_parity_test.cpp` | 固定真实帧 CPU/GPU 输出等价性回归 |
| `scripts/start_local_rtsp.ps1` | 按原帧率循环启动本地 RTSP |
| `scripts/stop_local_rtsp.ps1` | 校验 PID 与路径后停止测试服务 |
| `scripts/run_vehicle_rtsp_benchmark.ps1` | 基准、CPU/GPU/显存遥测和首尾趋势汇总 |
| `scripts/build_backend.ps1`、`CMakeLists.txt` | 构建 CUDA 源与新增测试/基准目标 |

此外，为使当前工作区已有的 M4 配置和已部署模型状态能通过原全量测试，修正了
`tests/model_delivery_contract_test.py`、`tests/vehicle_cascade_contract_test.py`
中的陈旧假设，并让 `web/camera-admin/app.js` 的等价 URL 表达满足现有字面合约；
这些调整不影响推理性能结果。

## 10. 30 分钟稳定性

45,000 个测量帧（另有 200 帧预热）的 RTSP 稳定性验收已连续运行
`1,813.640 s`（30 分 13.640 秒）并正常结束：

| 指标 | 结果 |
|---|---:|
| 完成测量帧 | 45,000 |
| 持续吞吐 | 24.812 FPS |
| 处理均值 / P95 | 8.845 / 10.789 ms |
| 采集状态 | running |
| open / reconnect | 1 / 0 |
| 目标进程 CPU 均值 / 峰值 | 2.800% / 12.669% |
| 目标进程 working set 均值 / 峰值 | 345.472 / 353.168 MiB |
| working set 首 / 尾 60 样本窗口 | 339.547 / 280.036 MiB |
| 全局 GPU 显存均值 / 峰值 | 2,936.794 / 3,098 MiB |
| GPU 显存首 / 尾 60 样本窗口 | 2,976.333 / 2,925.000 MiB |
| GPU 利用率均值 / 峰值 | 37.132% / 60% |
| GPU 温度峰值 | 51°C |

长稳期间没有崩溃、重连、显存持续增长或无界队列；采集配置使用大小固定为 1
的 latest-frame buffer。测量期处理了 45,000 个不同序列帧并输出结果，检测到
341 个序列间隙（0.758%）。该数量与 30 分钟内约 30 次 60 秒 MP4
`-stream_loop` 边界的约 11.4 帧/边界、最大 369.7 ms 源等待相吻合，属于本地
循环推流容器重开后的节拍追赶，而不是模型主动跳帧或队列积压。采集器原有的
`dropped_frames` 会在 latest 指针非空后对每次覆盖都累加，不能代表消费者漏帧；
本报告因此使用实际消费序列间隙作为判据。

Windows WDDM 下 `nvidia-smi --query-compute-apps` 对本进程显存返回 `N/A`，
无法可靠提供单进程显存；全局显存会结合测试前已存在进程进行解释。基准为计算
P50/P95 会写入一个预留上限为 45,000 的 `Sample` 数组（每项 128 字节，最多
约 5.49 MiB），因此 working set 的同量级、有界增长属于测量数据落页，不视为
生产 CUDA pipeline 泄漏。

证据：

- `reports/gpu-pipeline-optimization/rtsp-soak-30m/benchmark.json`
- `reports/gpu-pipeline-optimization/rtsp-soak-30m/telemetry.jsonl`
- `reports/gpu-pipeline-optimization/rtsp-soak-30m/telemetry-summary.json`

## 11. 剩余瓶颈与建议

1. 实时 RTSP 下 TensorRT 推理约 `6.90 ms`，已成为最大的模型处理阶段；可在
   不改变模型语义的前提下进一步评估 CUDA Graph 和并发摄像头批处理。
2. GPU 后处理在 25 FPS 间歇调用时约 `0.35 ms`，比当前 CPU 后处理约
   `0.22 ms` 略高，但避免了 756 KB 全量 D2H，独立满载时整体仍有收益。多路流
   场景应重新评估 GPU NMS 的 block-level 并行实现。
3. 属性分类器输出极小，当前 CPU softmax 的绝对耗时很低；若后续属性批量成为
   瓶颈，优先把 crop resize/normalize 批处理融合到 CUDA，而非先迁移 tiny softmax。
4. 真正的下一阶段低拷贝方案是 NVDEC surface -> CUDA 预处理 -> TensorRT，
   需要先演进帧接口，不能仅给 FFmpeg 加 `-hwaccel cuda`。

## 12. 第二轮范围与最终结论

第二轮在不更改视频、检测/属性模型、FP16 精度、输入尺寸、置信度阈值
`0.25`、class-aware NMS 阈值 `0.45` 和输出规则的前提下，逐项检查并实测了
A～H。最终保留：I420 直传、单输出像素预处理 kernel、CUDA Graph、紧凑
mapped-host 检测输出以及属性批量 GPU 预处理。额外单帧双缓冲、重建 TensorRT
引擎、并行 bitmask NMS 和 NVDEC surface 没有达到真实负载下的采用条件，均未
进入生产默认路径。

固定 25 FPS RTSP 的完整检测+真实检测框裁剪+属性分类 A/B 中，3 次独立运行
中位数为：

| 指标 | 兼容基线 | 第二轮最终组合 | 变化 |
|---|---:|---:|---:|
| 端到端 mean | 21.616 ms | 16.370 ms | **-24.27%** |
| 端到端 P95 | 32.981 ms | 24.854 ms | **-24.64%** |
| 持续 FPS | 25.006 | 24.999 | -0.03%，输入封顶 |

因此同时超过 mean/P95 至少降低 15% 的停止条件；结论不是由文件直读或单算子
基准替代得出。原始数据在
`reports/gpu-pipeline-optimization/round2/full-pipeline-ab/comparison.json`。
作为独立归因的无速率限制检测饱和基准也从 `363.795` 提高到 `466.851 FPS`
（+28.33%），mean/P95 分别降低 22.08%/23.55%。

## 13. 第二轮固定环境与新基线

环境由 `scripts/capture_gpu_pipeline_environment.ps1` 生成，完整版本和工件哈希
见 `reports/gpu-pipeline-optimization/round2/environment.json`：

| 项目 | 值 |
|---|---|
| CPU | AMD Ryzen 9 7945HX，32 logical processors |
| GPU | NVIDIA GeForce RTX 4080 Laptop GPU，12,282 MiB，CC 8.9 |
| Driver / 模式 | 581.80 / Windows WDDM |
| CUDA | 13.3，NVCC 13.3.73 |
| TensorRT | 10.16.1.11（runtime code v101601） |
| cuDNN | 配置的 CUDA/TensorRT runtime 中未安装，项目未链接 |
| OpenCV | 4.12.0 |
| FFmpeg | 2024-12-27 git full build，含 CUDA/CUVID/NVDEC |
| 构建 | Release，Ninja，MSVC 19.51，`/O2 /Ob2 /DNDEBUG`，CUDA arch 89 |
| 视频 SHA-256 | `27dfac5d82a857852b9533b3821a96777d2ec5f370c6740d005359cd8e1e5086` |
| 检测 ONNX SHA-256 | `a2655b27c9c1905f1ffbc3e16cb08e9800cca549d74cecc01d1b0bf47c504b93` |
| 检测 engine SHA-256 | `bdba13a99bea49c1bac448665fa102572bddf791a51c929b642020f586d53fb8` |
| 属性 engine SHA-256 | `eab83cc6e66133af5d61672e121465311ef914423dea54a2c32a6041ed98f5d6` |

所有正式短测均为 200 帧预热、2,000 帧测量、3 次独立运行取中位数；RTSP
来自同一 60 秒 MP4，以 `-re -stream_loop -1 -c:v copy` 在本机 25 FPS 循环
发布。系统中原有 `four_stage_worker` 占用 GPU，所有检查点均保留这一共享环境，
未改功耗或锁频。

本轮修改前的新 RTSP 检测基线为 mean `13.991 ms`、P95 `21.011 ms`、
`24.916 FPS`，CPU `5.115%`、RSS `346.368 MiB`、全局 GPU `31.786%`、
全局显存 `2400.573 MiB`。无实时限速饱和基线为 mean `2.747 ms`、P95
`2.912 ms`、`363.795 FPS`。证据分别位于 `round2/rtsp-baseline` 和
`round2/saturation-baseline`。

## 14. 第二轮 A～H 检查点

### A. I420 直接进入 GPU（保留）

- `FrameEnvelope` 新增带所有权的 I420 三平面描述和按需 `bgrImage()`；生产
  FFmpeg process reader 直接写入有界三缓冲 I420 ring，不再在采集线程执行
  `cv::cvtColor(I420->BGR)`。
- 检测请求新增兼容的 `I420ImageView`。CUDA kernel 融合 BT.601 limited-range
  I420->RGB、nearest letterbox、normalize 和 NCHW；BGR/CPU 路径仍可回退。
- pinned staging 和 device source buffer 按容量持久复用。1280x720 每帧 H2D
  从 `2,764,800` 降为 `1,382,400` 字节。
- 相对 B 检查点的 RTSP 中位数：mean `14.250 -> 12.869 ms`（-9.70%）；
  host staging `0.145 -> 0.122 ms`（-15.73%）；GPU preprocess kernel
  `0.749 -> 0.602 ms`（-19.61%）；H2D `0.961 -> 0.493 ms`（-48.64%）；
  H2D+preprocess `1.855 -> 1.218 ms`（-34.33%）。该隔离 run 的 P95 有噪声
  上升，最终组合正式 A/B 的 P95 则显著下降。

### B. CUDA 预处理 kernel（保留）

- 将旧的“一个线程写一个通道”改为一个线程处理一个输出像素：坐标、源索引和
  源像素只计算/读取一次，再写三个连续 NCHW plane，padding 也一次写三面。
- 饱和检查点 kernel `0.08264 -> 0.06547 ms`（-20.77%），preprocess
  `0.18168 -> 0.16423 ms`（-9.61%），H2D+preprocess `0.45888 ->
  0.43528 ms`（-5.14%），总 mean -1.90%，吞吐 +1.94%。
- `uchar4` 对三字节 BGR 和 I420 分平面会产生跨像素/跨平面非对齐读取；texture
  object 还会改变现有整数取样与 BT.601 舍入语义，并增加每帧/每分辨率绑定状态。
  在当前一次取样/像素且内存带宽并非主瓶颈时不保留。FP32 engine 输入也使
  `half2` 必须额外转换或重建 engine，不能作为同语义单变量检查点。

### C. CUDA Graph（保留）

- 固定地址后捕获 I420 H2D/kernel -> TensorRT enqueue -> GPU 后处理 ->
  mapped-host 紧凑输出。首次走传统路径预热，再实例化和复用 graph。
- graph key 包含输入元素数、源尺寸和像素格式；变化时销毁重建。捕获或实例化
  失败自动回到传统 stream 路径。事件计时移到 graph 外，避免 WDDM 下捕获
  event node 的 `invalid argument`。
- 饱和 mean `2.656 -> 2.141 ms`（-19.40%），P95 `2.816 ->
  2.226 ms`（-20.93%），吞吐 `376.235 -> 466.851 FPS`（+24.09%）；
  三轮共 6,000 个测量帧全部使用 graph，fallback 为 0。
- RTSP 相对前一检查点 mean `14.024 -> 11.613 ms`（-17.20%），P95
  `22.127 -> 20.482 ms`（-7.43%），实时吞吐不变。

### D. 多 stream 与有界缓冲（保留已有架构，不增加单帧 GPU slot）

- 生产采集为有界 latest-only queue，I420 reader 使用三个具备共享所有权检查的
  固定 buffer；两个 inference worker 各自拥有 runner、TensorRT context、CUDA
  stream 和 persistent buffer。配置仍支持一个 worker 的 single-stream 回退。
- 新增无静默丢帧的单/双 runner 饱和基准；每 stream 200+2,000 帧，顺序交替。
  单 stream 中位数 `468.544 FPS`，双 stream 总吞吐 `481.497 FPS`，仅
  `+2.765%`，`silent_drops=0`。
- 同一 runner 的同步返回契约要求当前帧紧凑结果才能继续，增加 frame slot 会扩大
  生命周期和串帧风险但没有可靠吞吐空间，故未把额外 GPU 双缓冲并入生产。跨
  摄像头仍由独立 stream 并发，帧序、时间戳和结果所有权保持一一对应。

### E. TensorRT engine 与 profiler（候选未采用）

- 用同一 ONNX/FP16/1x3x960x960 构建候选：optimization level 5、4 GiB
  workspace、timing cache、max auxiliary streams 2、detailed layer profile。
  原 ONNX 为静态 shape，显式 min/opt/max profile 被 TensorRT 正确拒绝，因此
  移除虚假的动态 profile 后重建。
- `trtexec --useCudaGraph --noDataTransfers`：吞吐 `388.420 -> 415.531
  qps`（+6.98%），GPU mean `2.465 -> 2.289 ms`（-7.17%），P95
  `2.950 -> 2.833 ms`（-3.97%）。应用饱和路径吞吐仅 +6.74%、mean
  -6.31%、P95 -4.73%。
- 候选仍通过 200 帧精确一致性和生产 graph 捕获，但增益低于采用阈值且引入两条
  auxiliary stream，所以不覆盖生产 engine。无代表性 INT8 校准集/cache，明确
  未构造 INT8 收益。
- detailed profiler 的生产热点包含 output transpose/div-mul、reshape copy 和
  attention/conv fusion；候选热点转为 Myelin SiLU/move-reshape/concat-reduce。
  构建日志、layer info/profile、参数、hash 和比较见 `round2/tensorrt-engine`。

### F. GPU 后处理与 NMS（紧凑回传保留，并行 NMS 撤销）

- 最终实现使用 mapped pinned host 输出；单 kernel 写数量和实际紧凑检测，删除
  “先 D2H 数量、host sync、再 D2H payload”的中途同步。饱和 D2H `0.08997
  -> 0.00438 ms`（-95.13%），总 mean -1.44%，吞吐 +1.45%。
- 实作并验证了上限 1,024 候选的 class-aware block bitmask NMS，超过上限精确
  回退 greedy；低 8 候选时 kernel 慢 15.4%、wall 慢 15.6%，实际饱和链路
  mean `2.141 -> 2.373 ms` 且 FPS `466.851 -> 421.100`，故完整撤销。
- 合成 800 密集候选时 bitmask wall 快约 94%，证明实现本身有效；但生产视频每帧
  最终仅约 8 个框。1,200 候选使用精确 fallback，性能基本持平。相同分数使用
  CPU `stable_sort` 与 CUB anchor/index tie-break 对齐，200 帧输出完全一致。
- segmented NMS 在当前低候选数会增加分段/launch；EfficientNMS 不存在于已部署
  静态 engine，加入 plugin 会改变 engine 输出契约。两者均不优于保留 greedy。

### G. 属性分类 GPU 批处理（保留）

- 将不同尺寸 crop 打包到可复用 pinned buffer，描述符和像素各一次 H2D；持久
  device buffer 上一个线程处理一个属性输出像素，直接完成 resize、BGR/RGB、
  normalize 和 batch NCHW。属性 runner 拥有独立 CUDA stream；CPU resize
  回退和极小的 CPU softmax 保留。
- 16 图 batch、200 batch 预热、2,000 batch x3 交替正式测试：preprocess
  `14.794 -> 1.416 ms`（-90.43%），总 `32.791 -> 21.258 ms`
  （-35.17%），吞吐 `473.513 -> 752.114 image/s`（+58.84%），H2D
  `12,582,912 -> 2,611,833` 字节/batch。
- 200 个不同 batch size（1～16）的 top-1 全部一致，最大置信度差 `0`。

### H. 真 NVDEC CUDA surface（真实实验，生产未启用）

- 使用 `h264_cuvid -> CUDA surface -> scale_cuda -> null`，明确没有
  `hwdownload`；与 software decode 交替执行 200+2,000 帧 x3。
- 中位数吞吐 `24.562 -> 24.497 FPS`（-0.26%）；归一化 FFmpeg CPU
  `0.178 -> 0.040%`（-77.55%，绝对仅 -0.138 个百分点）；峰值 RSS
  `104.88 -> 135.09 MiB`；平均全局显存约增加 `221.90 MiB`。
- 当前生产 process-pipe reader 只能传 host Y4M，无法传递 `AVHWFramesContext`
  和 surface 所有权；仓库/构建环境也没有 FFmpeg development libraries。因此
  接入必须新增 in-process libavcodec reader、设备上下文、surface pool/event、
  分辨率变化及重连生命周期管理。当前实时吞吐无收益且内存增加，生产保持软件
  FFmpeg 回退，仅保留实验脚本和后续接口设计证据。

## 15. 第二轮完整 RTSP 分阶段结果

`benchmark_vehicle_full_pipeline_ab.ps1` 交替启动独立进程；兼容侧强制按需
I420->BGR、关闭 graph、使用 CPU 属性预处理，优化侧启用 I420 直传、graph、
紧凑映射输出和属性 GPU batch。两侧仍共享 B/F 中已确认有效的实现，因而是偏保守
的最终比较。

| 三轮中位数 | 兼容基线 | 最终组合 |
|---|---:|---:|
| total mean / P50 | 21.616 / 20.002 ms | 16.370 / 15.105 ms |
| total P95 / P99 | 32.981 / 35.916 ms | 24.854 / 27.261 ms |
| RTSP decode/pull wait mean / P95 | 18.298 / 39.981 ms | 23.596 / 44.158 ms |
| CPU input prepare | 1.055 ms | 0.00011 ms |
| host staging | 0.150 ms | 0.096 ms |
| 检测 H2D | 2,764,800 B/frame | 1,382,400 B/frame |
| CUDA Graph | 0 | 2,000/2,000 frame，0 fallback |
| 属性 crop | 0.074 ms | 1.148 ms（含按需 BGR materialization） |
| 属性 preprocess | 3.705 ms | 0.231 ms |
| 属性 inference | 10.830 ms | 11.006 ms |
| 属性 postprocess | 0.0109 ms | 0.0107 ms |
| 属性 total | 16.235 ms | 12.128 ms |
| 测量期消费者序列缺口 | 0 | 0 |
| 目标进程 CPU | 3.899% | 3.179% |
| RSS | 393.307 MiB | 394.802 MiB |
| 全局 GPU 利用率 | 44.619% | 42.783% |
| 全局显存 mean / peak | 2960.446 / 2990 MiB | 2969.265 / 3008 MiB |

RTSP 等待随处理变快而增加，两项相加仍是相同 25 FPS 输入节拍；它不是解码回归。
正式三轮中两侧测量期均无消费者序列缺口，优化侧 P99、CPU 和显存没有明显回归。

## 16. 第二轮正确性与回归门槛

- 最终生产 engine 的固定 200 帧 CPU/GPU 对照：1,861 个检测，类别/数量一致，
  最小 IoU `1.0`，最大坐标差 `0 px`，最大分数差 `0`；CPU reference 最后一帧
  H2D `11,059,200` 字节，I420 graph 路径 `1,382,400` 字节。
- 属性 200 个可变 batch 对照：top-1 全部一致，最大 confidence delta `0`。
- Release 构建成功；28/28 CTest 通过，11 项需要 PostgreSQL/Redis/外部集成
  环境的测试按既有规则 skipped。`scripts/test_all.ps1` 的 Web Admin 与全部车辆
  Python contract suite 通过。
- CUDA 硬件测试、真实 TensorRT engine 测试、graph 6,000/6,000 帧和完整 RTSP
  6,000/6,000 帧均无 CUDA error、illegal access、graph fallback、死锁或串帧。

## 17. 第二轮复现命令

```powershell
$env:Path = "D:\GPU13.3\bin;D:\TensorRT-10.16.1.11\bin;" +
  "D:\libs\opencv\build\x64\vc16\bin;" +
  "E:\ffmpeg-7.1\ffmpeg-2024-12-27-git-5f38c82536-full_build\bin;$env:Path"

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_local_rtsp.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\capture_gpu_pipeline_environment.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_backend.ps1 `
  -BuildDir .\out\build\gpu-pipeline-round2-baseline
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_all.ps1 `
  -BuildDir .\out\build\gpu-pipeline-round2-baseline

.\out\build\gpu-pipeline-round2-baseline\vehicle_gpu_pipeline_parity_test.exe `
  (Get-Location).Path .\demo\rtsp_standard\output\vcas_rtsp_demo_60s.mp4
.\out\build\gpu-pipeline-round2-baseline\vehicle_attribute_gpu_benchmark.exe `
  (Get-Location).Path 2000 200 3 `
  .\reports\gpu-pipeline-optimization\round2\attribute-repro.json
.\out\build\gpu-pipeline-round2-baseline\vehicle_multistream_benchmark.exe `
  (Get-Location).Path .\demo\rtsp_standard\output\vcas_rtsp_demo_60s.mp4 `
  2000 200 3 .\reports\gpu-pipeline-optimization\round2\multistream-repro.json

powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\benchmark_vehicle_full_pipeline_ab.ps1 -Iterations 2000 `
  -WarmupFrames 200 -Runs 3
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\profile_vehicle_detection_engine.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\benchmark_nvdec_surfaces.ps1 -Iterations 2000 -WarmupFrames 200 -Runs 3
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run_vehicle_gpu_stability.ps1 -Iterations 45000 `
  -WarmupFrames 200 -RestartAfterSeconds 120
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_local_rtsp.ps1
```

## 18. 第二轮稳定性与剩余瓶颈

最终组合在同一进程中完成 200 帧预热和 45,000 个完整检测+属性测量帧，实际
`1,805.304 s`（30 分 5.304 秒）。第 2 分钟仅停止并重启本 Goal 状态文件中经
PID/可执行文件核验的 FFmpeg publisher；MediaMTX 和 benchmark 进程未重启。
长测同时覆盖约 30 次 60 秒视频循环边界，并反复复用 I420 三缓冲、pinned/
device buffer、CUDA Graph 和属性 batch buffer。

| 稳定性指标 | 结果 |
|---|---:|
| 完成测量帧 | 45,000 |
| 检测 / 属性结果 | 409,831 / 409,802 |
| 持续吞吐 | 24.927 FPS |
| total mean / P50 | 15.845 / 14.992 ms |
| total P95 / P99 | 23.954 / 27.270 ms |
| CUDA Graph / fallback | 45,000 / 0 |
| capture open / reconnect / state | 2 / 1 / running |
| 实际消费者序列缺口 | 94（0.209%） |
| 目标 CPU mean / max | 3.343% / 9.811% |
| RSS mean / max | 407.252 / 415.984 MiB |
| RSS 首 / 尾 60 样本窗口 | 389.326 / 394.379 MiB（+5.052） |
| 全局 GPU mean / max | 39.310% / 76% |
| 全局显存 mean / max | 3038.034 / 3867 MiB |
| 全局显存首 / 尾窗口 | 3543.717 / 2685.417 MiB（-858.300） |
| GPU 温度 max | 69 C |

94 个序列缺口包含容器循环边界和一次约 5 秒受控断流；重连后序号继续递增，
没有结果串帧或无界积压。`capture.dropped_frames` 是旧的 latest 指针“发生覆盖”
计数：指针不会在消费后清空，因此即使消费者逐序列读取也会累加，不能代表真实
漏帧；验收使用 benchmark 自己记录的 `consumer_sequence_gaps`。RSS 首尾增长
约 5 MiB 与预留 45,000 项统计样本数组的有界落页量相符。Windows WDDM 的全局
显存还包含用户原有 GPU 进程，窗口下降说明至少没有持续全局增长。原始数据和
验收摘要位于 `round2/stability-30m-reconnect`。

当前最终剩余瓶颈是属性 TensorRT inference（约 `11.006 ms`），而不是属性
preprocess（约 `0.231 ms`）。后续最值得尝试的是在不改变属性输出语义的前提下
重建/剖析属性 engine，以及在引入 FFmpeg dev libraries 后做真正的 in-process
NVDEC surface -> 检测/属性预处理原型。检测并行 NMS、额外双 stream 和本轮检测
engine 重建都已用数据证明不值得重复。

## 19. 第二轮证据索引

| 范围 | 结构化证据 |
|---|---|
| 总验收摘要 | `round2/final-summary.json` |
| 环境、版本、参数、hash | `round2/environment.json` |
| 本轮 RTSP / 饱和基线 | `round2/rtsp-baseline`、`round2/saturation-baseline` |
| B 单输出像素 kernel | `round2/rtsp-preprocess-pixel-kernel`、`round2/saturation-preprocess-pixel-kernel` |
| A I420 direct | `round2/rtsp-i420-direct` |
| F mapped compact D2H | `round2/rtsp-mapped-compact-d2h`、`round2/saturation-mapped-compact-d2h` |
| C CUDA Graph | `round2/rtsp-cuda-graph`、`round2/saturation-cuda-graph` |
| D 单/双 stream | `round2/multistream-saturation-alternating/benchmark.json` |
| E TensorRT build/profile | `round2/tensorrt-engine/profile-report.json`、`comparison.json`、layer JSON、build/profile logs |
| F 低/密集并行 NMS | `round2/postprocess-nms-low-dense/benchmark.json`、`round2/saturation-parallel-nms/benchmark.json` |
| G 属性 batch | `round2/attribute-gpu-preprocess-alternating/benchmark.json` |
| H NVDEC surface | `round2/nvdec-surfaces/benchmark.json` 及逐 run progress/log |
| 完整 RTSP A/B | `round2/full-pipeline-ab/comparison.json` 及六个独立 run 目录 |
| 30 分钟+重连 | `round2/stability-30m-reconnect` |
| 最终全量测试与 200 帧 parity | `round2/final-validation` |

以上路径均相对 `reports/gpu-pipeline-optimization`。其中 TensorRT detailed layer
profile 是本轮等价 profiler 关键证据；没有仅凭源代码或理论推断性能。

## 20. 第二轮交付文件与回滚

| 文件 | 第二轮用途 |
|---|---|
| `include/business/camera_frame_types.h` | I420 带所有权帧、惰性兼容 BGR |
| `src/business/ffmpeg_process_capture_reader.cpp` | 直接 Y4M/I420 三缓冲发布、重连和度量 |
| `src/business/rtsp_capture_reader.cpp`、`src/business/frame_artifact_writer.cpp` | 兼容惰性 BGR 消费 |
| `include/server/vehicle_model_contract.h`、`src/server/vehicle_model_contract.cpp` | `I420ImageView` 与请求契约 |
| `include/server/vehicle_cuda_preprocess.h`、`src/server/vehicle_cuda_preprocess.cu` | I420/BGR 融合单像素 kernel、分阶段 graph enqueue |
| `include/server/vehicle_cuda_postprocess.h`、`src/server/vehicle_cuda_postprocess.cu` | mapped pinned 紧凑结果和确定性 greedy NMS |
| `include/server/vehicle_cuda_attribute_preprocess.h`、`src/server/vehicle_cuda_attribute_preprocess.cu` | 属性 packed pinned batch CUDA 预处理 |
| `include/server/vehicle_tensorrt_adapters.h`、`src/server/vehicle_tensorrt_adapters.cpp` | persistent buffer、CUDA Graph、检测/属性接入与 telemetry |
| `include/server/model_runner.h`、`src/server/model_runner.cpp` | I420 请求路由 |
| `src/server/camera_inference_pool.cpp`、`src/server/camera_algorithm_processor.cpp`、`src/server/people_flow_inference_worker.cpp` | 帧所有权和生产消费者兼容 |
| `tests/vehicle_tensorrt_benchmark.cpp` | RTSP/文件、完整属性、compatibility、分阶段与资源基准 |
| `tests/vehicle_gpu_pipeline_parity_test.cpp` | 200 帧检测等价性和 graph invariant |
| `tests/vehicle_attribute_gpu_benchmark.cpp` | 属性 200 帧等价性与正式 A/B |
| `tests/vehicle_multistream_benchmark.cpp` | 无静默丢帧的单/双 stream 正式对照 |
| `scripts/benchmark_vehicle_full_pipeline_ab.ps1` | 交替完整链路 A/B 和 compact median 汇总 |
| `scripts/profile_vehicle_detection_engine.ps1` | 可复现 engine build、timing cache、detailed profiler |
| `scripts/benchmark_nvdec_surfaces.ps1` | 真 CUDA surface、无 hwdownload 的正式对照 |
| `scripts/capture_gpu_pipeline_environment.ps1` | 环境、编译参数和工件 hash |
| `scripts/restart_local_rtsp_publisher.ps1` | 仅重启经 PID/path 核验的目标 publisher |
| `scripts/run_vehicle_gpu_stability.ps1` | 45,000 帧、受控重连和自动验收 |
| `scripts/build_backend.ps1`、`CMakeLists.txt` | CUDA 源和 benchmark/test 目标 |

回滚不依赖替换模型：检测 runner 可分别关闭 `use_gpu_preprocess`、
`use_gpu_postprocess`、`use_cuda_graph`；属性 runner 可关闭
`use_gpu_preprocess`；`inference_workers=1` 保留单 stream 模式；不带 I420 的调用
自动使用 BGR 兼容路径。候选检测 engine 从未覆盖 `engines/vehicle-det-v1.engine`，
NVDEC 和并行 bitmask NMS 也未留在生产默认路径。

## 21. 第三轮范围与结论

第三轮从第二轮最终版本继续，保持视频、模型、阈值、输出语义和 25 FPS 实时
输入不变，主要处理属性 TensorRT、无 BGR 的 device-I420 ROI、属性 CUDA Graph、
跨帧并发、多路调度、内存同步以及完整业务输出。最终生产组合为：

- 检测后保留同一帧的 device I420 数据，属性 ROI 直接在 GPU 上完成采样、缩放、
  YUV->RGB、归一化和 NCHW 输出；正常路径不再构造整帧 BGR 或逐目标 `cv::Mat`。
- 检测和属性均保留安全的 TensorRT enqueue-only CUDA Graph；属性 descriptor、
  preprocess 和输出 copy 扩大 capture 范围的候选均未采用。
- 生产属性模型和输出 taxonomy 不变；重建 engine 与拆分分支候选因真实 ROI
  等价性失败而未替换生产 engine。
- 多路默认使用两个有界的直接属性 execution context，不默认启用全局动态合批，
  也不默认启用跨帧异步结果派发。
- 车辆快照改为一个 worker、容量为二的有界异步 JPEG writer；队列满时生产者
  阻塞，不静默丢任务，`vehicle_analytics.async_snapshots=false` 可单独回滚。

完整相同业务工作量的正式 25 FPS RTSP 三轮中位数中，兼容路径与最终路径分别为
`43.824 ms` 和 `17.866 ms`，mean 降低 `59.23%`；P95 从 `64.228 ms`
降至 `27.087 ms`（`-57.83%`），P99 从 `72.685 ms` 降至 `34.515 ms`
（`-52.51%`）。优化路径维持 `24.828 FPS`，capture-to-result mean 从
`60.784 ms` 降至 `24.411 ms`。无 25 FPS 限速的完整业务饱和测试中，吞吐从
`21.857 FPS` 提升到 `39.742 FPS`（`+81.83%`）。

## 22. 第三轮环境、基线和固定口径

结构化环境证据位于 `round3/environment.json`：Windows 11/WDDM、RTX 4080
Laptop 12,282 MiB、驱动 581.80、CUDA 13.3、TensorRT 10.16.1.11、
OpenCV 4.12.0。检测 engine、属性 engine 和测试视频 SHA256 分别为
`bdba13a...53fb8`、`eab83cc6...8f5d6`、`27dfac5d...1e5086`。
正式测试始终保留用户原有 server/worker，共享 GPU 状态没有为优化结果清场。

第三轮开始时重新建立的生产 RTSP 检测+真实 ROI 属性基线为：

| 指标 | 第三轮新基线 |
|---|---:|
| mean | 17.042 ms |
| P50 | 15.709 ms |
| P95 | 24.802 ms |
| P99 | 27.780 ms |
| FPS | 24.920 |
| capture-to-result mean / P99 | 25.699 / 40 ms |

该基线用于检查已有第二轮能力是否仍成立。第三轮最终 A/B 进一步加入 tracker、
属性聚合、逐帧 JSON 序列化、绘制和每 12 帧一次的 JPEG，因此性能验收使用同一
脚本内的 compatibility/optimized 对照，避免拿不同业务范围直接相减。所有正式
A/B 每个模式预热 200 帧、测量 2,000 帧、运行三次、交替顺序并取三轮中位数；
离线文件只用于饱和容量和完全相同内容的业务等价性，不能替代 RTSP 结果。

## 23. 属性 TensorRT、真实 batch 与候选矩阵

真实饱和负载 2,000 帧的属性 batch 直方图如下。每帧的有效 batch 语义保持不变，
没有为凑 bucket 增加业务结果：

| batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 帧数 | 0 | 0 | 30 | 110 | 104 | 219 | 253 | 274 |
| batch | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| 帧数 | 128 | 164 | 228 | 186 | 80 | 95 | 92 | 37 |

`scripts/profile_vehicle_attribute_engine.ps1` 对生产 engine、dynamic aux stream
0/2/4，以及 static batch 1/2/4/8/16 使用同一 ONNX、FP16、builder level 5、
4 GiB workspace 和同一 timing cache 建立矩阵。生产 engine 的纯计算 mean 在
batch 1/2/4/8/16 分别为 `1.975/2.698/4.390/9.414/21.985 ms`。逐层结果表明：
batch 1 的最大单层是 body head GEMM；batch 4～16 的时间分散在两个独立
ConvNeXt-Tiny 分支的大量 MatMul、GELU、归一化融合和 reformat 层，没有一个
可单独搬移的微小 CPU 算子能消除主体成本。

dynamic-aux2 在应用检查点中相对 production-I420 的 mean/P95 一度改善
`4.60%/9.14%`，但 220 个真实 RTSP batch、2,458 个 ROI 的最终 gate 出现一个
body top-1 不一致，最大置信度偏差 `0.342458`，因此候选未安装。按依赖裁出的
body/color 双 engine 也在 1,773 个真实 ROI 中出现四个 body top-1 不一致，
最大偏差 `0.342844`，并行分支实现已撤销。生产属性 engine 及其 hash 保持不变。

完整矩阵、构建参数、build log、layer info、layer profile、times JSON 和 timing
cache 位于 `round3/tensorrt-attribute`；采用/拒绝依据位于
`round3/attribute-engine-decision.json`。本机没有 Nsight Systems；等价时间线
证据由 TensorRT separate profile run、CUDA-event 分阶段计时、队列/序号/
capture-to-result telemetry 和每 batch Graph 计数共同组成，记录在
`round3/profiler-equivalent-evidence.json`。Nsight Compute 2026.2.1 可用，
但 layer profile 已确认成本分散在 TensorRT 内部两条 backbone，而非一个占端到端
10% 以上的自定义 kernel，因此没有用微型 kernel 结果替代应用结论。

## 24. 第三轮独立检查点

| 检查点 | 独立结果 | 决策 |
|---|---|---|
| host ROI -> device-I420 ROI | mean `-3.70%`，P95 `-1.37%`，属性 pixel H2D 正常路径消失 | 保留 |
| attribute enqueue Graph | 在 device ROI 上 mean `-2.70%`，P95 `-1.33%` | 保留 |
| device ROI + Graph | 相对 host/no-Graph mean `-6.29%`，P95 `-2.67%` | 保留 |
| descriptor+preprocess+enqueue+output Graph | 正确但 mean `+7.06%`、P95 `+32.08%` | 撤销 |
| enqueue+output-copy Graph | 真实 RTSP smoke 发生 `0xC0000005` | 撤销并保留失败日志 |
| dynamic-aux2 engine | 有性能信号，但真实 ROI top-1/置信度 gate 失败 | 拒绝 |
| body/color 分支并行 engine | 真实 ROI body top-1 gate 失败 | 拒绝并撤销运行时接入 |
| 有界跨帧异步结果派发 | 四路固定完成数吞吐 `-33.72%`，queue wait 与尾延迟增加 | 默认关闭 |
| 两个直接属性 context | 相对一个共享 scheduler：帧吞吐 `+16.30%`、属性吞吐 `+12.66%`、P95 `-12.79%` | 默认保留 |
| 全局/分片动态合批 | 一个共享和两个分片 scheduler 均慢于两个直接 context | 默认关闭，保留开关 |
| 有界异步快照 | mean `-24.71%`、P95 `-64.77%`、P99 `-67.71%`、FPS `+31.42%` | 默认保留 |

跨帧候选、调度器、内存和同步的汇总分别位于
`round3/async-pipeline/decision.json`、`round3/multi-route/decision.json` 和
`round3/memory-sync-audit.json`。稳态没有 `cudaDeviceSynchronize`；CUDA/pinned
buffer、event、ROI descriptor 和三槽 retained I420 pool 都按对象生命周期复用。
同步 runner 为返回具体结果仍需等待自己的 stream；实测的跨帧重叠因同 GPU 竞争
变慢，所以没有把“异步”本身当作收益。

## 25. 第三轮最终正式性能

完整 25 FPS RTSP（生产 reader/decode、检测、真实 ROI 属性、跟踪、JSON、绘制、
JPEG）的三轮中位数：

| 指标 | compatibility | optimized | 变化 |
|---|---:|---:|---:|
| mean | 43.824 ms | 17.866 ms | -59.23% |
| P50 | 42.471 ms | 16.406 ms | -61.37% |
| P95 | 64.228 ms | 27.087 ms | -57.83% |
| P99 | 72.685 ms | 34.515 ms | -52.51% |
| FPS | 21.437 | 24.828 | +15.82% |
| capture-to-result mean | 60.784 ms | 24.411 ms | -59.84% |
| capture-to-result P99 | 102 ms | 43 ms | -57.84% |
| consumer sequence gaps | 332 | 13 | -96.08% |

离线顺序解码的完整业务饱和测试使用 source-FPS 合成时间戳，确保 tracker 的时间
语义不随机器处理速度改变；六次运行都得到 18,203 次检测、18,201 次属性、
4,589,358 字节 JSON、167 张共 30,080,251 字节 JPEG，以及相同的
244/206/232 created/confirmed/exited 轨迹：

| 指标 | compatibility | optimized | 变化 |
|---|---:|---:|---:|
| mean | 41.753 ms | 19.380 ms | -53.58% |
| P50 | 39.774 ms | 18.531 ms | -53.41% |
| P95 | 65.348 ms | 29.002 ms | -55.62% |
| P99 | 78.034 ms | 36.729 ms | -52.93% |
| FPS | 21.857 | 39.742 | +81.83% |

所有最终运行中 snapshot writer 都满足 submitted=completed，失败数、最终队列深度、
active jobs、stale/version rejection 和检测/属性 Graph fallback 均为零；观测队列
峰值为 1，低于容量 2。

## 26. 多路、资源与最新瓶颈

四逻辑路、两个检测 worker、每模式每轮 2,000 个固定完成结果的三轮中位数：

| 属性执行方式 | frame FPS | attributes/s | capture P95/P99 |
|---|---:|---:|---:|
| 每路独立 4 context | 66.260 | 575.003 | 124.930 / 142.700 ms |
| 有界直接 2 context | 68.136 | 575.160 | 119.761 / 141.744 ms |
| 2 个分片 scheduler | 57.566 | 497.633 | 135.199 / 153.593 ms |
| 1 个共享 scheduler | 58.587 | 510.525 | 137.318 / 159.866 ms |

最终 RTSP 的 optimized 中位资源为进程 CPU `2.534%`、RSS `405.982 MiB`、
全局 GPU `39.581%`、全局显存 `2712.892 MiB`；相对相同业务 compatibility
分别变化 `-73.59%/+2.11%/+12.01%/+1.63%`。GPU 利用率上升对应更多有效
TensorRT 工作，CPU、P99、capture latency 和缺口同时下降；没有用锁频或功耗设置
计入收益。

optimized 中位 run 的属性 Graph 约 `9.674 ms`，占 end-to-end mean 的
`54.15%`；检测 Graph 约 `3.475 ms`，占 `19.45%`；快照调用约 `0.990 ms`，
占 `5.54%`。因此最新瓶颈仍是两个属性 ConvNeXt backbone 的计算，其同模型
engine、aux stream、静态 bucket、拆分分支和 Graph 范围均已有检查点。检测侧只有
在新的密集候选、多路或 TensorRT/CUDA 版本证明条件改变时才应重新评估。

## 27. 第三轮正确性和最终回归

最终正确性不是由性能计数间接推断：检测使用生产 engine 对固定视频逐帧比较
CPU/GPU 预处理和后处理，200 帧共 1,861 个检测，最小 IoU `1.0`、最大坐标差
`0 px`、最大分数差 `0`，且稳态 CUDA Graph 已命中。属性 enqueue-only Graph
使用 220 个真实 RTSP batch、1,418 个真实 ROI，比对结果 top-1 全部一致且无
fallback。CUDA 预处理契约还独立覆盖 padded Y/U/V stride、三槽 retained-device
frame pool 和分辨率/容量增长，避免只验证连续内存的理想输入。

最终 Release 构建在加载 amd64 Visual Studio Developer Command Prompt 后完成
52/52 个剩余 Ninja action。第一次直接调用 `cmake --build` 未加载 `VsDevCmd`，
`cl.exe` 因而找不到 MSVC 标准库头文件；该命令环境失败保留在
`round3/final-build.log`，相同构建目录的有效通过记录为
`round3/final-build-vcenv.log`，没有把它记为源码回归。

- CTest 使用临时 `postgres:17-alpine` 和 `redis:7-alpine`，显式启用破坏性测试
  数据库隔离开关：`30/30` 通过、`0` 失败、`0` skipped。容器结束后自动删除，
  端口 55433/56379 已释放。
- 10 个零依赖 Python/Web/车辆契约文件逐个按自身入口执行：`10/10` 通过。
- 部署中的两套真实 TensorRT engine 均完成 GPU 加载和推理；独立 CUDA contract
  与 200 帧 parity 均通过。
- 生产检测和属性 engine 都未被候选覆盖；所有性能数据来自相同 hash 的部署模型。

结构化汇总位于 `round3/build-validation.json`，原始输出位于同目录的
`ctest-final-integration.log`、`python-contracts-final.log`、
`final-*-engine.log`、`final-cuda-preprocess-contract.log` 和
`final-detection-parity-200.log`。

## 28. 45,000 帧稳定性、重连和有界性

最终组合持续运行 45,000 个测量帧，实际 `1,815.415 s`（30 分 15.415 秒）。
第 120 秒只重启由本轮状态文件管理且经过 PID/path 核验的 FFmpeg publisher；
MediaMTX 和 benchmark 不重启。测试覆盖约 30 次 60 秒源视频循环、一次受控断流/
重连、不同属性 batch、检测/属性 Graph、两个 execution context、三槽 device-frame
pool、快照队列及所有持久缓冲复用。

| 稳定性指标 | 结果 |
|---|---:|
| 完成测量帧 / 持续 FPS | 45,000 / 24.954 |
| 检测 / 属性结果 | 409,884 / 409,855 |
| total mean / P50 | 19.133 / 18.015 ms |
| total P95 / P99 / max | 30.459 / 39.027 / 115.248 ms |
| capture-to-result mean / P95 / P99 | 25.739 / 39 / 47 ms |
| 检测 Graph / fallback | 45,000 / 0 |
| 属性 Graph / fallback | 44,993 / 0 |
| capture open / reconnect / 最终状态 | 2 / 1 / running |
| 消费者序列缺口 | 65（0.144%） |
| business frames / JSON bytes | 45,000 / 105,743,619 |
| 快照 submitted / completed / failed | 3,750 / 3,750 / 0 |
| 快照最终 / 最大队列深度（容量） | 0 / 1（2） |
| 进程 CPU mean / max | 2.907% / 4.966% |
| RSS mean / max / 首尾窗口增长 | 429.368 / 447.957 / +13.659 MiB |
| GPU mean / max | 39.630% / 73% |
| 全局显存 mean / max / 窗口增长 | 2748.038 / 2773 / +33.100 MiB |
| GPU power mean / max | 69.356 / 125.530 W |
| GPU temperature mean / max | 65.498 / 71 C |

RSS 与全局显存窗口增长均低于预设 `64 MiB` 门槛，快照队列最终归零且峰值只有 1，
没有失败任务、stale/version rejection、Graph fallback、死锁或串帧。首个稳定性
尝试暴露 telemetry writer 与实时 reader 的文件共享冲突，失败样本被隔离保存；
脚本已改为单一 `FileStream/StreamWriter`、`FileShare.ReadWrite`、按 PID 轮询和精确
子进程清理。修复后的 reconnect smoke 与正式长测均通过。完整证据位于
`round3/stability-30m-reconnect/stability-summary.json`，脚本诊断位于
`round3/stability-runner-hardening.json`。

## 29. 后续完整方向图

以下优先级由第三轮最终 profile 决定；前两项因缺少外部输入而未伪造结果，第三项
以后属于新模型或新平台，必须单独版本化，不能计入本轮固定语义加速。

| 优先级 | 方向 | 最小进入条件 | 必须验证 |
|---:|---|---|---|
| 1 | in-process NVDEC CUDA surface | 安装与运行时 ABI 匹配、可供 x64 MSVC 链接的 libavformat/libavcodec/libavutil headers/import libs 与 CUDA hwcontext | 同一 RTSP 三轮 A/B、30 分钟重连、surface 引用/池有界性、CPU/P95/P99/RSS/VRAM；必须保留软件解码回滚且禁止 hwdownload 冒充零拷贝 |
| 2 | 属性 INT8 PTQ / INT8+FP16 | 冻结并批准与精确生产 ONNX/taxonomy 绑定的代表性图片/hash/标签 manifest | 总体和逐类 P/R、混淆矩阵、置信度漂移、困难切片、200 帧及更大 held-out real-ROI、应用 mean/P95/P99/FPS |
| 3 | 共享 backbone 属性模型或检测特征 ROIAlign | 冻结训练/验证集并保留当前双 ConvNeXt 工件回滚 | 完整模型发布 gate、轨迹稳定性、困难切片、真实 ROI 性能和 30 分钟稳定性 |
| 4 | 蒸馏、channel pruning、2:4 structured sparsity | 训练 checkpoint、许可数据、重训预算和目标 GPU tactic 支持 | 精度恢复后按真实 batch 3～16 分布 profile，再跑完整应用 gate |
| 5 | FP8 / INT4 混合精度 | 目标 GPU/TRT/operator 具备生产支持，且具有与 INT8 同等级校准/验证数据 | 敏感 normalization/head 保持 FP16/FP32 的独立候选；完整精度、置信度和应用性能 gate |
| 6 | 更新 TensorRT/CUDA 或迁移 Linux | 独立部署环境 | 同硬件隔离比较 tactic、reformat 和 WDDM 差异；重跑真实 ROI parity 与所有正式 A/B |
| 7 | 多 GPU / 按 camera-model 分片 / inference service | 生产路数超过已测两个直接 context 的单卡容量 | 单路公平性、总吞吐和归一化属性吞吐、queue P95/P99、VRAM、故障隔离、无饥饿；跨进程优先 CUDA IPC/shared memory |
| 8 | 条件性重做检测 top-k/NMS/多路 batching | 新 profile 显示检测已成为主瓶颈，或密集候选/版本条件改变 | 低/密候选、tie/order 语义、多路及饱和对照；不可原样重复已回归的 bitmask NMS |
| 9 | 业务策略减载：稳定轨迹复用、best-crop、相似 crop cache、质量延迟、采样/跳帧 | 独立业务规范允许少做工作 | 新鲜度、召回和业务准确率 gate；不得作为固定工作量第三轮加速宣称 |
| 10 | 生产可观测性扩展 | 安排 API/schema 兼容变更 | 将快照 submitted/completed/failed/depth/wait 暴露到 heartbeat/dashboard，并限制指标基数 |

当前外部阻塞的精确文件、接口和验收最低条件见 `round3/external-blockers.json`；
机器可直接开始的研究计划见 `round3/future-directions.json`。在没有新机制或新输入
时不要重复：扩大属性 Graph 到 preprocess/output（慢且一项崩溃）、同卡跨帧异步
派发（吞吐 -33.72%）、共享/分片全局 scheduler、aux2 rebuild、拆分双分支 engine、
小于 0.1 ms 的 CPU softmax/预处理微调，以及任何外部 raw pipe/hwdownload “零拷贝”。

## 30. 第三轮证据、复现和回滚

| 范围 | 结构化证据 |
|---|---|
| 最终总摘要 | `round3/final-summary.json` |
| 环境、版本、工件 hash | `round3/environment.json` |
| 所有检查点与采用/拒绝 | `round3/checkpoint-summary.json` |
| 属性 TRT 矩阵与逐层 profile | `round3/tensorrt-attribute`、`round3/attribute-engine-decision.json` |
| profiler 等价证据与真实 batch | `round3/profiler-equivalent-evidence.json` |
| device ROI / attribute Graph | `round3/device-graph-ab`、`round3/attribute-graph-real-roi-parity.json` |
| 多路调度与异步派发 | `round3/multi-route/decision.json`、`round3/async-pipeline/decision.json` |
| 快照 writer | `round3/snapshot-dispatch-ab/decision.json` |
| 正式饱和 / 正式 RTSP | `round3/saturation-final/decision.json`、`round3/rtsp-final/decision.json` |
| 30 分钟稳定性 | `round3/stability-30m-reconnect/stability-summary.json` |
| 构建、CTest、Python、硬件终验 | `round3/build-validation.json` 及其日志 |
| 外部阻塞与后续方向 | `round3/external-blockers.json`、`round3/future-directions.json` |
| 文件边界 | `round3/modified-files.json` |

核心复现顺序：

```powershell
$env:Path = "D:\GPU13.3\bin;D:\TensorRT-10.16.1.11\bin;" +
  "D:\TensorRT-10.16.1.11\lib;D:\libs\opencv\build\x64\vc16\bin;" +
  "E:\ffmpeg-7.1\ffmpeg-2024-12-27-git-5f38c82536-full_build\bin;$env:Path"

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_local_rtsp.ps1 `
  -StatePath .\runtime\rtsp-test\round3-state.json
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\capture_gpu_pipeline_environment.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\profile_vehicle_attribute_engine.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\benchmark_vehicle_round3_device_graph_ab.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\benchmark_vehicle_snapshot_dispatch_ab.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\benchmark_vehicle_full_pipeline_ab.ps1 -Iterations 2000 `
  -WarmupFrames 200 -Runs 3
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\scripts\run_vehicle_gpu_stability.ps1 -Iterations 45000 `
  -WarmupFrames 200 -RestartAfterSeconds 120
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_local_rtsp.ps1 `
  -StatePath .\runtime\rtsp-test\round3-state.json
```

运行时回滚是分层的：`vehicle_inference.compatibility_mode=true` 回退完整 GPU 组合；
`vehicle_analytics.async_snapshots=false` 只回退异步 JPEG；属性可单独关闭 device ROI、
GPU preprocess 或 enqueue Graph；`vehicle_analytics.attribute_contexts=1` 可收缩上下文；
`vehicle_analytics.dynamic_batching=true` 仅用于重新实验 scheduler；
`analysis.async_result_dispatch=false` 保持本轮通过的同步结果路径。生产 engine 不需
替换即可完成上述回滚。第三轮变更与既有脏工作区的边界见
`round3/modified-files.json`，没有重置用户文件、格式化无关源码或停止用户服务。
