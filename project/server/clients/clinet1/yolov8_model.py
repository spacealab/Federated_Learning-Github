import torch
from ultralytics import YOLO

def train_local_model(data_path):
    """
    Train the YOLOv8 model with local data using CPU.
    """
    model = YOLO("yolov8n.pt")  # Load the default YOLOv8 model
    model.to("cpu")  # Force the model to use CPU
    model.train(data=data_path, epochs=5, imgsz=640, device="cpu")  # Train the model on CPU
    return {k: v.cpu().data.numpy() for k, v in model.model.state_dict().items()}

def update_model_weights(global_weights, model_path="yolov8n.pt"):
    """
    Update the model weights with global weights using CPU.
    """
    model = YOLO(model_path)
    model.to("cpu")  # Force the model to use CPU
    model.model.load_state_dict({k: torch.tensor(v) for k, v in global_weights.items()})
    return model