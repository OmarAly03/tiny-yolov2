import cv2
import numpy as np

class BoundingBox:
    """Class to store a bounding box's properties."""
    def __init__(self, x, y, w, h, class_id, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id
        self.score = score
        
    def get_rect(self):
        """Returns (x_min, y_min, x_max, y_max) as floats"""
        x_min = self.x - self.w / 2
        y_min = self.y - self.h / 2
        x_max = self.x + self.w / 2
        y_max = self.y + self.h / 2
        return (x_min, y_min, x_max, y_max)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1. / (1. + np.exp(-x))

def softmax(x):
    """Compute softmax values for a set of scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two boxes."""
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1.get_rect()
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2.get_rect()
    
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return inter_area / union_area

def non_max_suppression(boxes, iou_threshold):
    """Apply Non-Maximum Suppression (NMS)"""
    boxes.sort(key=lambda b: b.score, reverse=True)
    
    suppressed_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        suppressed_boxes.append(current_box)
        
        boxes = [
            box for box in boxes 
            if box.class_id != current_box.class_id or 
               iou(current_box, box) < iou_threshold
        ]
        
    return suppressed_boxes

def decode_output(net_output, anchors, num_classes, obj_threshold):
    """
    Decodes the raw output from the YOLO model.
    net_output shape: (125, 13, 13)
    """
    grid_h, grid_w = net_output.shape[1:3]
    num_anchors = len(anchors)
    
    net_output = net_output.reshape((num_anchors, num_classes + 5, grid_h, grid_w))
    net_output = np.transpose(net_output, (2, 3, 0, 1))
    
    boxes = []
    
    for y in range(grid_h):
        for x in range(grid_w):
            for a in range(num_anchors):
                data = net_output[y, x, a]
                
                tx, ty, tw, th, objectness = data[:5]
                class_scores = data[5:]
                
                objectness_score = sigmoid(objectness)
                
                if objectness_score < obj_threshold:
                    continue
                
                cx = x
                cy = y
                
                center_x = (cx + sigmoid(tx))
                center_y = (cy + sigmoid(ty))
                
                width = anchors[a][0] * np.exp(tw)
                height = anchors[a][1] * np.exp(th)
                
                class_probs = softmax(class_scores)
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]
                
                final_score = objectness_score * class_score
                
                if final_score < obj_threshold:
                    continue

                boxes.append(BoundingBox(
                    center_x, center_y, width, height, class_id, final_score
                ))
                
    return boxes

def scale_and_draw_boxes(image, boxes, input_shape, labels):
    """
    Scales boxes from network input size to original image size
    and draws them on the original image.
    """
    img_h, img_w = image.shape[:2]
    net_h, net_w = input_shape
    grid_h, grid_w = 13, 13
    
    scale_x = img_w / grid_w
    scale_y = img_h / grid_h
    
    for box in boxes:
        x = box.x * scale_x
        y = box.y * scale_y
        w = box.w * scale_x
        h = box.h * scale_y
        
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        x_max = int(x + w / 2)
        y_max = int(y + h / 2)
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w - 1, x_max)
        y_max = min(img_h - 1, y_max)
        
        label = labels[box.class_id]
        score = box.score
        
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        text = f"{label}: {score:.2f}"
        
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x_min, y_min - text_h - baseline), (x_min + text_w, y_min), (255, 0, 0), -1)
        
        cv2.putText(image, text, (x_min, y_min - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image