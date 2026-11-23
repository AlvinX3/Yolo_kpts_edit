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
│   └── vitpose-base-simple/      ← whole HuggingFace folder (config.json, model.safetensors, ...)
├── rtdetr/
│   └── rtdetr_r50vd_coco_o365/   ← whole HuggingFace folder, person detector for ViTPose
└── openpose/                     ← not supported yet (see below)
```

press "Estimate" and pick a model in the dialog.

| pose model | who finds the people | set in the dialog |
|---|---|---|
| YOLO pose | the YOLO model itself (its own boxes) | detector box is greyed out |
| ViTPose | RT-DETR from `models/rtdetr/` | Person detector: `[RT-DETR] ...` (default) |
| ViTPose | YOLO from `models/yolo/` (person boxes only, its keypoints are ignored) | Person detector: `[YOLO] ...` |

## ViTPose (new)

pose estimation can now use ViTPose as well as YOLO.

- download a ViTPose model folder from HuggingFace and put the whole folder in `models/vitpose/`,
  e.g. https://huggingface.co/usyd-community/vitpose-base-simple
- ViTPose only predicts keypoints inside person boxes, so it also needs a person detector.
  choose it under "Person detector" in the dialog: an RT-DETR folder from `models/rtdetr/`
  (e.g. https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365, RT-DETRv2 folders work too)
  or a YOLO model from `models/yolo/` (only its person boxes are used).
- YOLO pose models don't use the detector: they find people with their own boxes.
- output is the same COCO 17-keypoint pose JSON as YOLO, so editing and saving work the same way.
- ViTPose Confidence is a heatmap score, not the same scale as YOLO Confidence;
  the 0.5 "reliable" threshold may flag different frames.

## OpenPose

**OpenPose is NOT supported yet.** models put in `models/openpose/` are ignored and will not show up in the dialog.
the folder is only reserved for future support.
