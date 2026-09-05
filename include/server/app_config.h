#pragma once

#include <map>
#include <string>
#include <vector>

#include "server/camera_profile.h"

namespace yolo11_server {

    struct ServerSection {
        std::string host = "0.0.0.0";
        int port = 8080;
        int threads = 4;

        // Production default: HTTP server is a producer/query process.
        // Keep sync detection disabled unless you really want the server process to load TensorRT.
        bool enable_sync_detect = false;

        // Request body guard. 0 means disabled.
        int max_body_size_mb = 16;
    };

    struct ModelSection {
        std::string type = "detect";
        std::string engine_path = "./engines/yolo11n.engine";
        std::string labels_path = "./labels/coco.txt";
        int gpu_id = 0;
        bool use_gpu_postprocess = false;

        // Phase 17: classification top-k output size. Used only when type=cls.
        int cls_topk = 5;
    };

    struct OutputSection {
        // Generate and store a result image for async/sync detection.
        bool save_result_image = true;

        // Local paths are still kept for compatibility and debugging.
        // In production split mode, async workers prefer Redis image bytes first.
        std::string input_dir = "./temp/input";
        std::string output_dir = "./output";
        int jpeg_quality = 90;
    };

    struct VideoSection {
        // Phase 13: video-file async inference. Keep this false for normal image detect/OBB servers.
        bool enabled = false;
        std::string input_dir = "./temp/video/input";
        std::string output_dir = "./temp/video/output";

        // Request/file limits for video upload. 0 means disabled.
        long long max_video_bytes = 256LL * 1024LL * 1024LL;

        // Worker updates Redis progress every N frames.
        int progress_update_interval_frames = 30;

        // Limit for early experiments. 0 means process all frames.
        int max_process_frames = 0;

        // Output writer settings. mp4v works on most Windows OpenCV builds.
        std::string output_extension = ".mp4";
        std::string output_fourcc = "mp4v";
        double fallback_fps = 25.0;
    };


    struct StreamSection {
        // Phase 14: live stream / RTSP / camera task management.
        bool enabled = false;

        // camera / file / rtsp. Phase 14.0 can start with camera or file to validate lifecycle.
        std::string default_source_type = "camera";
        int default_camera_id = 0;
        std::string default_file_path;

        // Phase 19.2: RTSP credentials live only in the referenced process
        // environment variable. YAML contains profile metadata, never a URI.
        std::string camera_profiles_path;
        std::string default_camera_profile;

        // Latest snapshot is overwritten periodically, not every frame.
        std::string snapshot_dir = "./runtime/output/streams";
        int snapshot_interval_frames = 5;
        int target_fps = 10;

        // Phase 14.5: stream stability controls.
        // max_no_frame_count is the short local tolerance before a reconnect/fail decision.
        int max_no_frame_count = 30;
        bool enable_reconnect = true;
        int reconnect_max_attempts = 3;
        int reconnect_delay_ms = 1000;

        // 0 means unlimited. Useful for smoke tests and long-run guards.
        int max_runtime_seconds = 0;

        // Phase 14.5: if a non-terminal stream has no live worker heartbeat
        // or no latest update for this long, HTTP can mark it failed and
        // release the active-stream guard. This prevents stale running tasks
        // after a worker crash from blocking the next /stream/start forever.
        int stale_timeout_ms = 30000;

        // JPEG quality for snapshot output.
        int jpeg_quality = 90;
    };

    // Phase 19.3: production RTSP capture controls. RTSP decoding runs in a
    // dedicated capture thread and exposes only the newest decoded frame to
    // the inference loop, preventing latency from growing with a frame queue.
    struct CaptureSection {
        std::string backend = "ffmpeg";
        int open_timeout_ms = 5000;
        int read_timeout_ms = 3000;
        int stale_frame_timeout_ms = 5000;
        int buffer_size = 1;
        std::string transport = "tcp";
        bool latest_frame_only = true;
        int warmup_frames = 10;
        int reconnect_max_attempts = 0;  // 0 means unlimited.
        int reconnect_initial_delay_ms = 500;
        int reconnect_max_delay_ms = 10000;
        int status_update_interval_ms = 1000;
        bool allow_backend_fallback = false;
    };

    struct CameraHubSection {
        bool enabled = true;
        int max_active_hubs = 4;
        int idle_grace_ms = 5000;
        bool require_ffmpeg_backend = true;
        int status_update_interval_ms = 1000;
    };

    struct CameraTaskDefaultsSection {
        int frame_interval_ms = 1000;
        std::string output_mode = "latest";
        int jpeg_quality = 90;
        int max_width = 0;
        int max_height = 0;
        int retention_days = 7;
        int max_saved_frames = 100000;
        bool analysis_enabled = false;
        double target_infer_fps = 5.0;
        std::string algorithm_profile;
        std::vector<std::string> algorithms;
        std::string callback_profile;
    };

    struct CameraTaskStoragePolicySection {
        // 0 disables the corresponding guard for backwards-compatible test
        // and embedded deployments.
        long long max_archive_bytes = 0;
        long long min_free_bytes = 0;
        int high_watermark_percent = 85;
        int critical_watermark_percent = 95;
        int pressure_cleanup_batch_size = 1000;
        std::string backup_dir = "./runtime/backups";
        int backup_retention_count = 7;
    };

    struct CameraTasksSection {
        bool enabled = false;
        std::string postgres_dsn_env = "YOLO11_POSTGRES_DSN";
        std::string output_dir = "./runtime/output/camera_frames";
        std::string admin_ui_dir = "./web/camera-admin";
        std::string command_stream_key = "yolo:stream:camera-frame";
        std::string consumer_group = "yolo11_camera_frame_group";
        int max_active_runs = 4;
        int writer_threads = 2;
        int writer_queue_capacity = 32;
        int writer_queue_capacity_per_run = 8;
        int status_ttl_seconds = 604800;
        int lease_ttl_seconds = 30;
        int lease_refresh_seconds = 5;
        int stale_run_timeout_ms = 30000;
        int retention_sweep_interval_seconds = 60;
        int retention_batch_size = 500;
        std::string admin_token_env = "YOLO11_CAMERA_TASK_ADMIN_TOKEN";
        CameraTaskDefaultsSection defaults;
        CameraTaskStoragePolicySection storage;
        std::string config_error;
    };

    struct AnalysisSection {
        bool enabled = true;
        // Fixed at process startup. Each worker owns exactly one model runner.
        int inference_workers = 2;
        int model_init_timeout_ms = 120000;
        // Experimental cross-frame overlap. Disabled by default because the
        // round-3 four-route hardware A/B reduced aggregate throughput.
        bool async_result_dispatch = false;
        int result_queue_capacity = 2;
        std::vector<std::string> supported_algorithms{
            "vehicle_detection",
            "vehicle_attribute"
        };
    };

    struct VehicleAnalyticsServiceSection {
        bool enabled = true;
        std::string config_path = "./config/vehicle_analytics.yaml";
        std::string model_registry_path =
            "./models/manifests/model_registry.v1.json";
        int observation_retention_days = 7;
        int snapshot_fps = 2;
        int jpeg_quality = 90;
        // One attribute context per inference worker avoids serializing dense
        // multi-camera traffic. Set to 1 for the low-VRAM rollback path.
        int attribute_contexts = 2;
        // Experimental cross-camera aggregation. Round-3 measurements keep it
        // disabled because queueing outweighed the larger batches.
        bool dynamic_batching = false;
        // Periodic analysis JPEG encoding runs on a bounded CPU worker so its
        // long tail overlaps GPU inference. Set false for synchronous rollback.
        bool async_snapshots = true;
        int snapshot_writer_threads = 1;
        int snapshot_queue_capacity = 2;
    };


    struct CallbackProfileSection {
        bool enabled = true;
        // Endpoint and HMAC secret values are resolved only inside the Worker.
        // YAML stores environment-variable names, never the values themselves.
        std::string url_env;
        std::string hmac_secret_env;
        bool allow_insecure_http = false;
    };

    struct CallbackDeliverySection {
        bool enabled = false;
        int poll_interval_ms = 250;
        int request_timeout_ms = 5000;
        int lease_timeout_ms = 30000;
        int max_attempts = 8;
        int initial_backoff_ms = 1000;
        int max_backoff_ms = 300000;
        int request_body_limit_bytes = 1048576;
        int response_body_limit_bytes = 4096;
        std::map<std::string, CallbackProfileSection> profiles;
        std::string config_error;
    };


    struct RedisSection {
        bool enabled = true;
        std::string host = "127.0.0.1";
        int port = 6379;
        std::string password;
        int db = 0;
        std::string stream_key = "yolo:stream:detect";
        std::string consumer_group = "yolo11_group";
        std::string consumer_name = "worker_1";
        int block_ms = 1000;

        // Default TTL used by task status/meta/result JSON when more specific TTLs are not set.
        int ttl_seconds = 1800;
        int task_ttl_seconds = 1800;

        // Redis Binary Image Storage controls.
        // input_image_ttl_seconds can be shorter because workers delete input image after XACK when enabled.
        int input_image_ttl_seconds = 600;
        int result_image_ttl_seconds = 1800;
        long long max_image_bytes = 5LL * 1024LL * 1024LL;
        long long max_result_image_bytes = 5LL * 1024LL * 1024LL;
        bool delete_input_after_done = true;

        // Optional Redis memory guard for /ready and async submit. 0 means disabled.
        long long max_redis_used_memory_mb = 2048;

        // Keep Redis Stream length bounded. 0 means disabled.
        // Uses approximate trimming: XTRIM stream MAXLEN ~ stream_max_len.
        long long stream_max_len = 10000;

        // Reclaim messages delivered to a worker but not XACKed.
        bool enable_pending_reclaim = true;
        long long pending_min_idle_ms = 60000;

        // Phase 8.5: runtime metrics stored in Redis.
        bool metrics_enabled = true;
        int metrics_recent_window_seconds = 60;
        int metrics_ttl_seconds = 3600;
    };

    struct LoggingSection {
        bool enabled = true;
        std::string log_dir = "./logs";
        std::string level = "info";
        bool console = true;
        bool file = true;
        int flush_interval_sec = 3;
        int max_file_size_mb = 50;
        int max_files = 5;
    };

    struct WorkerSection {
        // For all-in-one debug mode only. Production yolo11_server should use enabled=false.
        bool enabled = false;

        int worker_num = 1;
        int min_alive_workers = 1;
        std::string consumer_name_prefix = "worker_";
        bool log_task_done = true;

        // Phase 15: worker capability/resource declaration.
        // These fields are written into the heartbeat so /ready, /workers and
        // /metrics can reason about model/task isolation before introducing
        // real multi-GPU automatic scheduling.
        std::string worker_group;       // e.g. image_detect_gpu0 / video_detect_gpu0 / stream_detect_gpu0
        std::string worker_kind;        // image / video / stream
        std::string task_kind;          // image_async / video_file / live_stream
        std::string stream_type;        // redis_stream / long_running_stream
        int max_concurrency = 1;        // descriptive capacity, not an automatic scheduler yet

        // Phase 8: heartbeat written by yolo11_worker.exe / InferenceWorker.
        bool heartbeat_enabled = true;
        int heartbeat_interval_ms = 3000;
        int heartbeat_ttl_seconds = 15;
    };

    // Phase 10.5: optional model profile entry used by legacy multi-model YAML configs.
    // The current process still activates one model profile at a time; this keeps the
    // Server/Worker split simple while making detect/obb configuration explicit.
    struct ModelProfile {
        std::string name;
        std::string type;
        std::string engine_path;
        std::string labels_path;
        int gpu_id = 0;
        bool use_gpu_postprocess = false;
        int cls_topk = 5;
        std::string stream_key;
        std::string consumer_group;
        std::string consumer_name_prefix;
        int worker_num = 1;
        int min_alive_workers = 1;
    };

    struct AppConfig {
        ServerSection server;
        ModelSection model;
        OutputSection output;
        VideoSection video;
        StreamSection stream;
        CaptureSection capture;
        CameraHubSection camera_hub;
        CameraTasksSection camera_tasks;
        AnalysisSection analysis;
        VehicleAnalyticsServiceSection vehicle_analytics;
        CallbackDeliverySection callbacks;
        RedisSection redis;
        LoggingSection logging;
        WorkerSection worker;

        // Phase 19.2 camera registry loaded from stream.camera_profiles_path.
        std::map<std::string, CameraProfile> camera_profiles;
        std::string camera_profiles_error;

        // Optional multi-model profile registry.
        // active_model is empty for legacy single-model YAML.
        std::string active_model;
        std::map<std::string, ModelProfile> model_profiles;

        static AppConfig loadFromYaml(const std::string& yaml_path);
    };

}  // namespace yolo11_server
