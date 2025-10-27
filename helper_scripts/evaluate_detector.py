from ultralytics import YOLO

def cross_evaluation():
    models = ['models/yolo12CoastlineDrone.pt', 'models/yolo12SeaDroneSee.pt']
    data_paths = ['data/CoastlineDrone-OD.yaml', 'data/SeaDroneSee-OD.yaml']

    for model in models:
        yolo_model = YOLO(model)
        
        for data_path in data_paths:
            # Run detection on a folder, with restricted classes
            print(f"\n\nRUNNING {model} on {data_path} TRAIN\n\n")
            metrics = yolo_model.val(data=data_path, split='train', classes=[0, 1, 2], save=False)

            print(f"\n\nRUNNING {model} on {data_path} VAL\n\n")
            metrics = yolo_model.val(data=data_path, split='val', classes=[0, 1, 2], save=False)

def evaluate_merged_dataset():
    models = ['models/yolo12s_CoastlineDrone.pt', 'models/yolo12s_SeaDroneSee.pt']
    data_path = 'data/MergedDataset-OD.yaml'

    for model in models:
        yolo_model = YOLO(model)
        
        # Run detection on a folder, with restricted classes
        print(f"\n\nRUNNING {model} on {data_path} TRAIN\n\n")
        metrics = yolo_model.val(data=data_path, split='train', save=False)

        # Run detection on a folder, with restricted classes
        print(f"\n\nRUNNING {model} on {data_path} VAL\n\n")
        metrics = yolo_model.val(data=data_path, split='val', save=False)

evaluate_merged_dataset()
