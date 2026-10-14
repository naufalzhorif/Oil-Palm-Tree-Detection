CONFIG = {
    "dataset": {
        "ann_file": "_annotations.coco.json", # annotation json file name
        "train": "dataset/train", # train dataset path
        "valid": "dataset/valid", # valid dataset path
        "dataloader": {
            "batch_size": 8,
            "num_workers": 1,
        },
    },
    "training": {
        "epochs": 30, # epochs
        "checkpoint_dir": "checkpoints", # checkpoint directory
        "optimizer": {
            "lr": 0.0001, # learning rate
        },
        "scheduler": {
            "step_size": 5,
            "gamma": 0.1,
        },
    },
    "inference": {
        "model_path": "./checkpoints/run_280926_1013/last.pt", # trained model path
        "model_type": "torchvision", # model type (eg. torchvision, ultralytics)
        "image_path": "dataset/test/SEPT_1KDP_RGB_FINAL.tif", # image path
        "tile_size": 512, # image tile size
        "conf_threshold": 0.5, # confidence
        "overlap": 0.2, # overlap (in %)
        "postprocess": "NMS", # postprocessing (eg. nms, greedy_nms, nmm)
        "save_result_dir": "detection_result", # save result directory
        "category_names": ["palmTree"], # change category_names with your category
    },
}