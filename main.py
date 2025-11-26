import onnxruntime as ort
import cv2
import numpy as np
import os
import glob
from utils import decode_output, non_max_suppression, scale_and_draw_boxes

# --- Configuration ---
MODEL_PATH = 'tinyyolov2.onnx' 
IMAGES_FOLDER = './images/'    
OUTPUT_FOLDER = './output_images/' 

# --- Constants ---
LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

ANCHORS = [
    [1.08, 1.19],
    [3.42, 4.41],
    [6.63, 11.38],
    [9.42, 5.11],
    [16.62, 10.52]
]

def process_single_image(session, image_path, output_path, input_name, output_name, net_h, net_w):
    """Process a single image and save the result."""
    print(f"Processing: {image_path}")
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Pre-process
    image_resized = cv2.resize(original_image, (net_w, net_h))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32)
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(image_chw, axis=0)
    
    # Inference
    outputs = session.run([output_name], {input_name: input_tensor})
    output_data = outputs[0][0]
    
    # Post-process
    OBJECT_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.3
    
    boxes = decode_output(output_data, ANCHORS, len(LABELS), OBJECT_THRESHOLD)
    
    if boxes:
        final_boxes = non_max_suppression(boxes, NMS_THRESHOLD)
        print(f"  Found {len(final_boxes)} objects")
        
        output_image = scale_and_draw_boxes(
            original_image.copy(), final_boxes, (net_h, net_w), LABELS
        )
        cv2.imwrite(output_path, output_image)
        print(f"  Saved to: {output_path}")
    else:
        print(f"  No objects detected")
        cv2.imwrite(output_path, original_image)
    
    return True

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGES_FOLDER, extension)))
        image_files.extend(glob.glob(os.path.join(IMAGES_FOLDER, extension.upper())))
    
    if not image_files:
        print(f"No image files found in {IMAGES_FOLDER}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    net_h, net_w = input_info.shape[2], input_info.shape[3]
    if not isinstance(net_h, int):
        net_h, net_w = 416, 416
    
    print(f"Network Size: {net_h}x{net_w}")
    print("-" * 50)
    
    # Process images
    successful_count = 0
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"detected_{name}{ext}")
        
        print(f"[{i}/{len(image_files)}] ", end="")
        
        if process_single_image(session, image_path, output_path, 
                              input_info.name, output_info.name, net_h, net_w):
            successful_count += 1
    
    print("-" * 50)
    print(f"Processing complete! {successful_count}/{len(image_files)} images processed")
    print(f"Results saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()