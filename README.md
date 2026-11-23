it just a yolo pose editer

> pip install -r requirements.txt

## Models

put models in the models folder, one sub-folder per framework:
```
models/
├── yolo/
│   ├── yolo26x-pose.pt
│   └── yolo11x-pose.pt
├── vitpose/
│   └── vitpose-base-simple/   ← whole HuggingFace folder (config.json, model.safetensors, ...)
└── openpose/                  ← not supported yet (see below)
```

press "Estimate" and pick a model in the dialog.

## ViTPose (new)

pose estimation can now use ViTPose as well as YOLO.

- download a ViTPose model folder from HuggingFace and put the whole folder in `models/vitpose/`,
  e.g. https://huggingface.co/usyd-community/vitpose-base-simple
- ViTPose only predicts keypoints inside person boxes, so it also needs a YOLO model
  (from `models/yolo/`) as the person detector. choose it under "Person detector" in the dialog.
- output is the same COCO 17-keypoint pose JSON as YOLO, so editing and saving work the same way.
- ViTPose Confidence is a heatmap score, not the same scale as YOLO Confidence;
  the 0.5 "reliable" threshold may flag different frames.

## OpenPose

**OpenPose is NOT supported yet.** models put in `models/openpose/` are ignored and will not show up in the dialog.
the folder is only reserved for future support.
