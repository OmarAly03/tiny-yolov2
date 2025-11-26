# TinyYoloV2

This project implements object detection using a pretrained TinyYOLOv2 ONNX model. The application processes all images in the `images/` directory and generates annotated outputs with bounding boxes in the `output_images/` folder using ONNX Runtime for efficient inference.

## Model Information

**Pre-trained Model:** TinyYOLOv2 ONNX format  
**Source:** [Hugging Face Model Repository](https://huggingface.co/webml/models-moved/blob/4ff2d9e89e61cd77eaaae2eae8b36d7ee50ce878/tinyyolov2-8.onnx)  
**Runtime:** ONNX Runtime for cross-platform inference

## Requirements

- **Python3**
- **Dependencies:**
  - `numpy==2.2.6`
  - `onnxruntime==1.22.1`
  - `opencv-python==4.12.0.88`

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/OmarAly03/tiny-yolov2.git
cd tiny-yolov2
```

2. **Set up virtual environment and install dependencies:**
```bash
python3 -m venv tyv2
source tyv2/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
tiny-yolov2/
├── images/                 # Input images for detection
│   ├── cat.jpg
│   ├── man_tv.jpg
│   ├── mark1.jpg
│   └── street.jpg
├── output_images/          # Generated output images with detections
├── main.py                 # Main inference script
├── utils.py                # Utility functions for preprocessing/postprocessing
├── tinyyolov2.onnx        # Pre-trained ONNX model file
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```