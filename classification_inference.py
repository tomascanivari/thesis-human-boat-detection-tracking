from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def classify_folder(folder_path: str, model_path: str = "models/ClassificationModels/fold0.pt"):
    model = YOLO(model_path)

    folder = Path(folder_path)
    image_paths = list(folder.glob("*.[jp][pn]g"))  # matches .jpg and .png

    for img_path in tqdm(image_paths):
        results = model.predict(source=str(img_path), verbose=False, device=0, half=False, stream=True)
        for r in results:
            predicted_class = int(r.probs.top1)

if __name__ == "__main__":
    classify_folder("datasets/ClassificationDataset/images/Coastline")