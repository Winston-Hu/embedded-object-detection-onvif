import time
from datetime import datetime
from pathlib import Path
import ctypes
import threading

import cv2
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_PATH = Path(r"D:/github_repos/embedded-object-detection-onvif/YOLO/ultralytics/weights/yolov8s.pt")

CAMERA_IP = "192.168.1.64"
CAMERA_PORT = 554
USERNAME = "admin"
PASSWORD = "1234qwer"
CHANNEL = 101

SHOW_WINDOW = True
SAVE_VIDEO = True
WINDOW_NAME = "YOLOv8 Hikvision RTSP Monitor"

IMGSZ = 640
CONF = 0.5
IOU = 0.45
DEVICE = "0"

RETRIES = 5
RETRY_DELAY = 2.0
DEFAULT_FPS = 25.0
DEBUG_EVERY_N_FRAMES = 30
WINDOW_MARGIN = 80
READ_TIMEOUT_SEC = 5.0


def build_rtsp_url() -> str:
    return f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{CAMERA_PORT}/Streaming/Channels/{CHANNEL}"


def build_output_video_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"detect_{timestamp}.mp4"


def open_stream(rtsp_url: str, retries: int = RETRIES, retry_delay: float = RETRY_DELAY) -> cv2.VideoCapture:
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            return cap
        cap.release()
        print(f"[WARN] RTSP open failed, retry {attempt}/{retries}...")
        time.sleep(retry_delay)
    raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")


def create_writer(output_path: Path, fps: float, frame_width: int, frame_height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_fps = fps if fps and fps > 1 else DEFAULT_FPS
    writer = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create output video: {output_path}")
    return writer


def get_model_device(model: YOLO) -> str:
    try:
        return str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return "unknown"


def print_runtime_debug(model: YOLO, frame, annotated_frame, frame_count: int) -> None:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(
        "[DEBUG] "
        f"frame={frame_count} "
        f"raw_shape={frame.shape} "
        f"annotated_shape={annotated_frame.shape} "
        f"device_arg={DEVICE} "
        f"model_device={get_model_device(model)} "
        f"cuda_available={cuda_available} "
        f"gpu_name={gpu_name}"
    )


def get_screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def resize_for_display(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame, 1.0

    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))
    display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    return display_frame, scale


class LatestFrameStream:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_capture_ts = 0.0
        self.last_read_ts = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.cap = open_stream(self.rtsp_url)
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, name="rtsp-latest-frame", daemon=True)
        self.thread.start()

    def _reader_loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            now = time.time()
            if not ok or frame is None:
                print("[WARN] Reader thread failed to read frame, reconnecting RTSP stream...")
                if self.cap is not None:
                    self.cap.release()
                self.cap = open_stream(self.rtsp_url)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.latest_capture_ts = now
                self.last_read_ts = now

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0, 0.0
            return self.latest_frame.copy(), self.latest_frame_id, self.latest_capture_ts

    def get_stream_properties(self):
        if self.cap is None:
            raise RuntimeError("Stream is not started.")
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return frame_width, frame_height, fps

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


def main() -> None:
    rtsp_url = build_rtsp_url()
    output_video_path = build_output_video_path()

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    print(f"[INFO] Torch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
        print(f"[INFO] CUDA device 0: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Requested inference device: {DEVICE}")

    print(f"[INFO] Opening RTSP stream: {rtsp_url}")
    stream = LatestFrameStream(rtsp_url)
    stream.start()

    frame_width, frame_height, fps = stream.get_stream_properties()
    print(f"[INFO] Stream size: {frame_width}x{frame_height}, FPS: {fps:.2f}")
    screen_width, screen_height = get_screen_size()
    preview_max_width = max(320, screen_width - WINDOW_MARGIN)
    preview_max_height = max(240, screen_height - WINDOW_MARGIN)
    print(f"[INFO] Screen size: {screen_width}x{screen_height}")
    print(f"[INFO] Preview max size: {preview_max_width}x{preview_max_height}")

    writer = None
    if SAVE_VIDEO:
        writer = create_writer(output_video_path, fps, frame_width, frame_height)
        print(f"[INFO] Saving annotated video to: {output_video_path}")

    frame_count = 0
    start_time = time.time()
    window_initialized = False

    try:
        while True:
            frame, frame_id, capture_ts = stream.get_latest()
            if frame is None:
                if time.time() - start_time > READ_TIMEOUT_SEC:
                    print("[WARN] Waiting for first RTSP frame...")
                time.sleep(0.005)
                continue

            loop_start = time.time()
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False,
            )
            infer_done = time.time()

            annotated_frame = results[0].plot()
            frame_count += 1

            if frame_count == 1 or frame_count % DEBUG_EVERY_N_FRAMES == 0:
                print_runtime_debug(model, frame, annotated_frame, frame_count)

            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
            infer_ms = (infer_done - loop_start) * 1000.0
            end_to_end_ms = (time.time() - capture_ts) * 1000.0 if capture_ts > 0 else -1.0
            frame_gap = max(0, stream.latest_frame_id - frame_id)
            cv2.putText(
                annotated_frame,
                f"FPS: {avg_fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Infer: {infer_ms:.1f} ms  E2E: {end_to_end_ms:.1f} ms  Dropped: {frame_gap}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated_frame)

            total_loop_ms = (time.time() - loop_start) * 1000.0
            if frame_count == 1 or frame_count % DEBUG_EVERY_N_FRAMES == 0:
                print(
                    "[PERF] "
                    f"frame={frame_count} "
                    f"source_frame_id={frame_id} "
                    f"infer_ms={infer_ms:.1f} "
                    f"loop_ms={total_loop_ms:.1f} "
                    f"e2e_ms={end_to_end_ms:.1f} "
                    f"avg_fps={avg_fps:.2f} "
                    f"dropped_frames={frame_gap}"
                )

            if SHOW_WINDOW:
                display_frame, display_scale = resize_for_display(
                    annotated_frame,
                    preview_max_width,
                    preview_max_height,
                )
                if not window_initialized:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, display_frame.shape[1], display_frame.shape[0])
                    print(
                        f"[INFO] Display frame size: {display_frame.shape[1]}x{display_frame.shape[0]} "
                        f"(scale={display_scale:.3f})"
                    )
                    window_initialized = True
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Quit requested, stopping.")
                    break
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video saved: {output_video_path}")
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
