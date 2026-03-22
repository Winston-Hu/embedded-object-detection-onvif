import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_PATH = Path(r"D:/github_repos/embedded-object-detection-onvif/YOLO/ultralytics/weights/yolov8s.pt")

CAMERA_IP = "192.168.1.100"
CAMERA_PORT = 554
USERNAME = "admin"
PASSWORD = "1234qwer"
CHANNEL = 101

SHOW_WINDOW = True
SAVE_VIDEO = True
WINDOW_NAME = "YOLOv8 Hikvision RTSP Monitor"

IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = "0"

RETRIES = 5
RETRY_DELAY = 2.0
DEFAULT_FPS = 25.0


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


def main() -> None:
    rtsp_url = build_rtsp_url()
    output_video_path = build_output_video_path()

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print(f"[INFO] Opening RTSP stream: {rtsp_url}")
    cap = open_stream(rtsp_url)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Stream size: {frame_width}x{frame_height}, FPS: {fps:.2f}")

    writer = None
    if SAVE_VIDEO:
        writer = create_writer(output_video_path, fps, frame_width, frame_height)
        print(f"[INFO] Saving annotated video to: {output_video_path}")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Failed to read frame, reconnecting RTSP stream...")
                cap.release()
                cap = open_stream(rtsp_url)
                continue

            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                verbose=False,
            )

            annotated_frame = results[0].plot()
            frame_count += 1

            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
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

            if writer is not None:
                writer.write(annotated_frame)

            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] Quit requested, stopping.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video saved: {output_video_path}")
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
