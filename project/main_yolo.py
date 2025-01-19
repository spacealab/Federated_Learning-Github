from ultralytics import YOLO

# Load a YOLO model
model = YOLO("./runs/detect/train6/weights/best.pt")

results = model("./VID_20250103181613.mp4", save=True, show=True)