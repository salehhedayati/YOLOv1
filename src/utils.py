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

import torch

def batch_iou(predictions, targets):
    """
    Compute Intersection over Union (IoU) for batches of bounding boxes.
    
    Args:
        predictions (torch.Tensor): Predicted bounding boxes of shape (N, 4), where N is the number of boxes.
                                    Each box is represented as [x, y, w, h], where (x, y) is the center coordinate
                                    of the box and (w, h) are the width and height.
        targets (torch.Tensor): Ground truth bounding boxes of shape (N, 4), where N is the number of boxes.
                                 Each box is represented as [x, y, w, h], where (x, y) is the center coordinate
                                 of the box and (w, h) are the width and height.
    
    Returns:
        torch.Tensor: Intersection over Union (IoU) for each pair of bounding boxes.
                      Returns a tensor of shape (N,) containing IoU values.
    """
    # Extract individual coordinates for predictions and targets
    pred_x, pred_y, pred_w, pred_h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    target_x, target_y, target_w, target_h = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    # Compute coordinates of top-left and bottom-right corners for predictions and targets
    pred_x1 = pred_x - pred_w / 2
    pred_y1 = pred_y - pred_h / 2
    pred_x2 = pred_x + pred_w / 2
    pred_y2 = pred_y + pred_h / 2

    target_x1 = target_x - target_w / 2
    target_y1 = target_y - target_h / 2
    target_x2 = target_x + target_w / 2
    target_y2 = target_y + target_h / 2

    # Compute areas of predictions and targets
    pred_area = pred_w * pred_h
    target_area = target_w * target_h

    # Compute coordinates of intersection rectangle
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    # Compute area of intersection rectangle
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Compute IoU
    iou = inter_area / (pred_area + target_area - inter_area + 1e-6)  # Add epsilon to avoid division by zero

    return iou

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

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