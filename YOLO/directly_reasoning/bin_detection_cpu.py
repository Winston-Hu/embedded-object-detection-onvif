from ultralytics import YOLO
import os


script_dir = os.path.dirname(__file__)
img_path = os.path.join(script_dir, "images", "bin.jpg")

# Load pretrained model (COCO dataset, includes trash can)
model = YOLO("best.pt")  # n=smallest, s/m/l/x are larger

# Run prediction on an image -> multiple pics with boxes info
results = model.predict(
    source=img_path,
    device="cpu",
    save=True,  # save: True -> bonding box
    conf=0.5
)

# Show detection results in console
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}, BBox: {xyxy}")

