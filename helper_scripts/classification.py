import torch
from ultralytics import YOLO



epochs = 5
folds = [5]


for fold in folds:

    # Clear memory
    torch.cuda.empty_cache()

    # Load a model
    model = YOLO("yolo12n-cls.yaml")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=f"MergedDataset/classification/fold{fold}", epochs=epochs, imgsz=640, save=True, batch=12, name=f"MergedDatasetFold{fold}", exist_ok=True)