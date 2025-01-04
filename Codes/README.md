A complete codebase and configuration files are provided to ensure result reproducibility.

The scripts have been optimized to accommodate various hardware environments. When using the code, the user needs to change its relative path.

The purposes of the code files in the "Codes" folder are as follows:

**YOLO/eval_yolo.py** generates json file according to the YOLO model.

**YOLO/transform.py** turns the origin data into the right format for YOLO model.

**YOLO/train.py** is the YOLO training code.

**YOLO/split.py** divides the training data into training set and validation set randomly.

**YOLO/coco.yaml** is the configuration file for YOLO.

**Faster R-CNN/Faster-RCNN-eval.py** generates json file according to the Faster R-CNN model.

**Faster R-CNN/Faster-RCNN-train.py** is the Faster R-CNN training code.