# Camera and Vehicle Runtime Operations

## Runtime invariant

- Run one `four_stage_worker` process (`worker.worker_num: 1`).
- Use one process-local FrameHub and one FFmpeg decode per Camera Profile.
- Camera extraction and vehicle inference consume the same immutable frames
  through independent subscriptions.
- PostgreSQL is the durable store; Redis carries commands, leases, heartbeats,
  and hot state; the filesystem stores JPEG artifacts.
- RTSP URIs, callback endpoints/secrets, PostgreSQL/Redis credentials, and the
  admin token are supplied only through environment variables.

## PostgreSQL bootstrap

```powershell
$secure = Read-Host "PostgreSQL password" -AsSecureString
$credential = [System.Net.NetworkCredential]::new("", $secure)
$env:YOLO11_POSTGRES_PASSWORD = $credential.Password
docker compose -f .\deploy\postgresql\compose.yaml up -d
$env:YOLO11_POSTGRES_DSN = "host=127.0.0.1 port=5432 dbname=vision_project user=vision_app password=$($credential.Password) sslmode=disable"
$credential = $null
$secure.Dispose()
```

Do not print or persist the DSN. Both Server and Worker read the environment
variable named by `camera_tasks.postgres_dsn_env`. Integration tests must use a
separate disposable database through `YOLO11_TEST_POSTGRES_DSN`; never point
that variable at production.

Apply the SQL files in numeric order for existing deployments. New repository
initialization uses the idempotent embedded schema and the same versioned
migrations under `db/postgresql/`.

## Build, test, and start

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_backend.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\test_all.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

`start_demo.ps1` prompts without echo for missing credentials and starts the
Worker and Server. The administration page is
`http://127.0.0.1:8087/camera-admin`.

## Camera lifecycle

All routes require `Authorization: Bearer <admin token>`.

| Operation | Route | Runtime effect |
|---|---|---|
| Create | `POST /api/v1/cameras` | Persist desired state and start one internal Run when enabled |
| Read | `GET /api/v1/cameras[/{camera_id}]` | Merge durable state with current Run/Hub state |
| Update | `PATCH /api/v1/cameras/{camera_id}` | Require ETag; stop/join the old generation and start its replacement |
| Delete | `DELETE /api/v1/cameras/{camera_id}` | Require ETag; stop the current generation and soft-delete the Camera |
| Start/stop | `POST .../{camera_id}/start` or `/stop` | Idempotent lifecycle control |
| History | `GET .../{camera_id}/runs` | Read-only internal Run history |

Camera Profiles are deployment-owned and read-only over HTTP. There are no
public Camera Task CRUD routes and no person-analysis compatibility routes.

## Callback delivery

Callback delivery is disabled by default. To enable the built-in
`backend_primary` profile, inject `YOLO11_CALLBACK_BACKEND_PRIMARY_URL` and
`YOLO11_CALLBACK_BACKEND_PRIMARY_SECRET` into both processes, then enable the
profile in both YAML files. Keep `allow_insecure_http=false` outside controlled
loopback tests.

The callback worker signs the exact request bytes with HMAC-SHA256, claims
outbox rows with a fenced lease, retries timeout/408/429/5xx responses with
bounded exponential backoff, and dead-letters terminal failures. Receivers must
verify timestamp/signature headers and deduplicate by the stable `event_id`.

Dead-letter recovery:

1. Query `GET /api/v1/operations/callbacks?status=dead_letter&limit=20`.
2. Inspect the sanitized status and correct the external dependency.
3. POST `/api/v1/operations/callbacks/{outbox_id}/replay` with the observed
   attempt as `If-Match` and an empty JSON object.
4. Confirm `202`, then observe `retry` to `delivered`.

## Acceptance checks

1. Confirm `/api/v1/ready` reports a fresh Worker heartbeat and ready vehicle
   inference/callback components.
2. Create a Camera, wait for `running`, and verify `latest-frame` returns JPEG.
3. Confirm the configured vehicle detection and attribute algorithms are active.
4. Verify Hub diagnostics show one source open for every shared Camera Profile.
5. PATCH with the latest ETag and confirm the old Run stops before its
   replacement becomes active.
6. Submit START twice and confirm the active Run ID is reused.
7. Query vehicle realtime/events and verify low-confidence attributes degrade
   to `unknown` instead of forced labels.
8. DELETE with the latest ETag and confirm the Pipeline subscription is reclaimed.

Use the benchmark, restart, reconnect, multi-camera, dead-letter, and soak
scripts under `scripts/` for hardware acceptance. Evidence is written below
ignored `reports/` directories and must never include credentials.

## Failure behavior

| Fault | Expected behavior |
|---|---|
| PostgreSQL unavailable | repository initialization fails closed; readiness stays false |
| Redis unavailable | durable definitions remain; mutations fail with sanitized errors; hot status becomes stale |
| RTSP disconnect | the shared Hub reconnects; all consumers observe the same source state |
| Queue/storage pressure | bounded queues drop and count work without blocking the Hub |
| Worker crash | lease-fenced recovery marks stale nonterminal Runs failed and restarts desired Cameras |
| Stale ETag | PATCH/DELETE returns 409 without changing the live generation |
| Callback retryable failure | durable outbox enters `retry` with bounded backoff |
| Callback terminal failure | durable outbox enters `dead_letter` |

## Backup and restore

With Server and Worker stopped and PostgreSQL environment variables set:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\backup_runtime.ps1 -Label scheduled
powershell -ExecutionPolicy Bypass -File .\scripts\restore_runtime.ps1 `
  -BackupPath .\runtime\backups\vision_runtime_<stamp>.zip -ConfirmRestore
```

Restore validates the archive, creates a pre-restore backup, and uses
`pg_restore --clean --if-exists`. JPEG output is excluded and must be backed up
separately when archive-image recovery is required.
