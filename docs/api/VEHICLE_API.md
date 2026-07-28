# Vehicle Event API

Base URL：`http://127.0.0.1:8087/api/v1`

所有接口都要求：

```http
Authorization: Bearer <YOLO11_CAMERA_TASK_ADMIN_TOKEN>
```

接口不返回 RTSP URI、数据库 DSN、回调地址、密钥或本机绝对路径。时间字段统一为 Unix epoch 毫秒。

## 实时轨迹

`GET /cameras/{camera_id}/vehicles/realtime`

响应来自注入到 `CameraTaskHttpController` 的 `VehicleRealtimeSnapshotReader`，不读取历史数据库。Reader 尚未接入 Worker 时仍返回 `200`，但 `available=false` 且 `items=[]`；Reader 执行失败返回 `503 VEHICLE_REALTIME_UNAVAILABLE`。

```json
{
  "success": true,
  "camera_id": "entrance_01",
  "available": true,
  "generated_at_ms": 1785200000000,
  "items": [
    {
      "camera_id": "entrance_01",
      "run_id": "run_01",
      "run_generation": 7,
      "track_id": 42,
      "state": "confirmed",
      "last_seen_at_ms": 1785200000000,
      "vehicle_class": {"label": "car", "confidence": 0.97},
      "attributes": {
        "body_type": {
          "label": "suv",
          "confidence": 0.91,
          "stable": true,
          "samples_used": 4
        },
        "color": {
          "label": "white",
          "confidence": 0.88,
          "stable": true,
          "samples_used": 4
        }
      }
    }
  ]
}
```

只返回与路径中 `camera_id` 一致的轨迹。摄像头不存在返回 `404 TASK_NOT_FOUND`。

## 历史事件

`GET /cameras/{camera_id}/vehicle-events`

查询参数：

- `occurred_from_ms`：可选，包含下界；默认不限制。
- `occurred_to_ms`：可选，包含上界；默认不限制。
- `limit`：默认 20，范围 1–200。
- `offset`：默认 0，范围 0–1,000,000。

当开始时间晚于结束时间时返回 `400 INVALID_TIME_RANGE`。结果按 `occurred_at_ms DESC, event_id DESC` 稳定排序。

```json
{
  "success": true,
  "camera_id": "entrance_01",
  "occurred_from_ms": null,
  "occurred_to_ms": null,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "schema_version": "1.0",
      "event_id": "ve_0123456789abcdef",
      "event_kind": "vehicle_passage",
      "camera_id": "entrance_01",
      "run_id": "run_01",
      "track_id": 42,
      "occurred_at_ms": 1785200000000,
      "vehicle_class": {"label": "car", "confidence": 0.97},
      "attributes": {
        "body_type": {
          "label": "suv",
          "confidence": 0.91,
          "stable": true,
          "samples_used": 4
        },
        "color": {
          "label": "white",
          "confidence": 0.88,
          "stable": true,
          "samples_used": 4
        }
      },
      "model_versions": {
        "detector": "vehicle_detector_v1",
        "attribute": "vehicle_attribute_v1",
        "labels": "vehicle-labels-v1",
        "config": "vehicle-analytics-m3"
      },
      "evidence": {
        "crop_quality": 0.93,
        "snapshot_url": "/api/v1/vehicle-events/ve_0123456789abcdef/snapshot"
      },
      "delivery": {"status": "pending"},
      "created_at_ms": 1785200000100
    }
  ]
}
```

## 事件详情

`GET /vehicle-events/{event_id}`

成功响应把与历史列表相同的事件对象放在 `event` 字段。事件不存在返回 `404 VEHICLE_EVENT_NOT_FOUND`。

## 事件快照

`GET /vehicle-events/{event_id}/snapshot`

成功返回 `Content-Type: image/jpeg` 和 `Cache-Control: no-store`。服务端只解析 Camera Task 输出根目录内的项目相对路径，并校验 JPEG SOI/EOI；空路径、越界路径、缺失文件或非法 JPEG 都返回 `404 VEHICLE_SNAPSHOT_NOT_READY`。

## 模型交付状态

`GET /models/status`

该接口只读 Model Registry，不探测或加载 TensorRT Engine。`ready=true` 要求 Artifact 的 `delivery_status` 为 `engine_validated` 或 `deployed`，并存在 64 位十六进制长度的 `engine_sha256` 字段。

```json
{
  "success": true,
  "enabled": true,
  "registry_version": "1.0",
  "labels_version": "vehicle-labels-v1",
  "all_ready": false,
  "items": [
    {
      "artifact_id": "vehicle_detector_v1",
      "role": "detection",
      "delivery_status": "planned",
      "backend": "tensorrt",
      "precision": "fp16",
      "onnx_sha256": null,
      "engine_sha256": null,
      "ready": false
    }
  ]
}
```

Registry 不可读或结构无效返回 `503 MODEL_STATUS_UNAVAILABLE`。车辆分析被禁用时返回 `enabled=false`、`all_ready=false` 和空列表。

## 回调

车辆事件与算法告警共用可靠 Callback Outbox。回调 Body 是落库时冻结的 `vehicle_passage` JSON，不包含查询专用的 `delivery` 字段；`Idempotency-Key` 和 `X-Event-Id` 均为 `event_id`。签名、重试、死信和人工重放语义与 [Camera API](../CAMERA_FRAME_TASK_API.md) 一致。
