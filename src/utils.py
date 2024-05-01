import torch

def iou(predictions, targets):
    box1_x1 = predictions[..., 0:1] - predictions[..., 2:3]/2
    box1_y1 = predictions[..., 1:2] - predictions[..., 3:4]/2
    box1_x2 = predictions[..., 0:1] + predictions[..., 2:3]/2
    box1_y2 = predictions[..., 1:2] + predictions[..., 3:4]/2
    box2_x1 = targets[..., 0:1] - targets[..., 2:3]/2
    box2_y1 = targets[..., 1:2] - targets[..., 3:4]/2
    box2_x2 = targets[..., 0:1] + targets[..., 2:3]/2
    box2_y2 = targets[..., 1:2] + targets[..., 3:4]/2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    a1 = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    a2 = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (a1 + a2 - intersection + 1e-6)

# Example data
predictions = torch.tensor([
    [0.5, 0.5, 0.4, 1.3],  # [x, y, w, h] format
    [0.6, 0.7, 0.3, 3.4],
    [0.3, 0.4, 0.5, 3.6]
])

targets = torch.tensor([
    [0.5, 0.5, 0.4, 2.3],  # [x, y, w, h] format
    [0.7, 0.6, 0.3, 5.4],
    [0.4, 0.3, 0.5, 9.6]
])

# Compute IoU
iou = iou(predictions, targets)
print("Intersection over Union (IoU):", iou)