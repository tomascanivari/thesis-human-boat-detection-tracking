import torch
from ultralytics import YOLO

# Clear memory
torch.cuda.empty_cache()

def fine_tune_coastlinev2(fold, resume):

    # === 1. Define Path & Arguments === #
    if resume:
        model_path = f'runs/detect/CoastlineV2BigDatasetFold{fold}/weights/last.pt'
        kwargs = {
            'resume': True
        }
    else:
        model_path = 'runs/detect/CoastlineV2Dataset/weights/best.pt'
        kwargs = {
            "data":      f"data/CoastlineV2BigDatasetFold{fold}.yaml",
            "epochs"     : 5,
            "device"     : "cuda",
            "imgsz"      : 640,
            "save"       : True,
            "batch"      : 12,
            "name"       : f"CoastlineV2BigDatasetFold{fold}",
            "exist_ok"   : True,
            "lr0"        : 0.001,
            "optimizer"  : 'SGD',
            "save_period": 1
        }

    # 2. === Load & Fine-Tune the Model === #
    model = YOLO(model=model_path)

    model.train(**kwargs)

###############
# Main Script #
###############

# Folds {'fold_number': ('completed', 'resume')}
folds = {
    0: (True, False),
    1: (True, False),
    2: (True, False),
    3: (True, False), 
    4: (True, False),
    5: (False, True)
}

# Process Each Fold
for fold, actions in folds.items():
    
    completed, resume = actions
    
    if completed:
        continue

    fine_tune_coastlinev2(fold, resume)





















# fine_tune = True

# dataset = "CoastlineV2"
# resume = True
# n_epochs = 200

# fold = 1

# if fine_tune:
    
#     fine_tuned_dataset = "CoastlineV2Big"
#     n_epochs = 5
#     if not resume:
#         # Load the model
#         model = YOLO(model = "runs/detect/CoastlineV2Dataset/weights/best.pt")


#         # Train the model
#         results = model.train(data=f"data/{fine_tuned_dataset}DatasetFold{fold}.yaml", epochs=n_epochs, imgsz=640, save=True, batch=12, name=f"{fine_tuned_dataset}DatasetFold{fold}", exist_ok=True, lr0=0.001, optimizer='SGD', save_period=1)

#     else:
#          # Load the last epoch of the model
#         model = YOLO(model = f"runs/detect/CoastlineV2BigDatasetFold{fold}/weights/last.pt")
        
#         # Resume the training
#         model.train(resume=True)


# if dataset == "CoastlineV2":
#     if not resume:
#         # Load the model
#         model = YOLO(model = "yolo12s.yaml")

#         # Train the model
#         results = model.train(data=f"data/{dataset}Dataset.yaml", epochs=n_epochs, imgsz=640, save=True, batch=12, name=f"{dataset}Dataset", exist_ok=True)
#     else:
#         # Load the last epoch of the model
#         model = YOLO(model = f"runs/detect/{dataset}Dataset/weights/last.pt")
        
#         # Resume the training
#         model.train(resume=True)

# else:
#     if not resume:
#         # Load the model
#         model = YOLO(model = "yolo12s.yaml")

#         # Train the model
#         results = model.train(data=f"data/{dataset}DatasetFold{fold}.yaml", epochs=n_epochs, imgsz=640, save=True, batch=12, name=f"{dataset}DatasetFold{fold}", exist_ok=True)

#     else:
#         # Load the last epoch of the model
#         model = YOLO(model = f"runs/detect/{dataset}DatasetFold{fold}/weights/last.pt")
        
#         # Resume the training
#         model.train(resume=True)


