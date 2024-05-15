import numpy as np
import torch
from collections import Counter, deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import utils

from heapq import nlargest

def IoU(predictions, target):
    # predictions/target (N, S, S, 4)
    p_x1 = predictions[..., 0:1] - predictions[..., 2:3]/2
    p_y1 = predictions[..., 1:2] - predictions[..., 3:4]/2
    p_x2 = predictions[..., 0:1] + predictions[..., 2:3]/2
    p_y2 = predictions[..., 1:2] + predictions[..., 3:4]/2
    t_x1 = target[..., 0:1] - target[..., 2:3]/2
    t_y1 = target[..., 1:2] - target[..., 3:4]/2
    t_x2 = target[..., 0:1] + target[..., 2:3]/2
    t_y2 = target[..., 1:2] + target[..., 3:4]/2
    
    x1 = torch.max(p_x1, t_x1)
    y1 = torch.max(p_y1, t_y1)
    x2 = torch.min(p_x2, t_x2)
    y2 = torch.min(p_y2, t_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # equal to 0 for cells with no target
    
    a1 = abs((p_x2 - p_x1) * (p_y2 - p_y1))
    a2 = abs((t_x2 - t_x1) * (t_y2 - t_y1))
    
    return intersection / (a1 + a2 - intersection + 1e-6)

def nms(bboxes, iou_threshold, cs_threshold):
    # bboxes (list) : [[c, cs, x, y, w, h] x S*S]
    assert type(bboxes) == list
    # max_values = []
    # sorted_data = sorted(bboxes.copy(), key=lambda x: x[1], reverse=True)  # Sort in descending order
    # sorted_deque = deque(sorted_data)
    # max_values.extend(nlargest(2, sorted_deque, key=lambda x: x[1]))
    bboxes = [box for box in bboxes if box[1] > cs_threshold]
    
    if not bboxes:
        # all bboxes have confidence score less than threshold
        return []
    
    sorted_bboxes = deque()
    sorted_bboxes.extend(sorted(bboxes, key=lambda x:x[1], reverse=True))
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = sorted_bboxes.popleft()
        
        bboxes = [
            box
            for box in sorted_bboxes
            if box[0] != chosen_box[0]
            or IoU(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold
        ]
        
        bboxes_after_nms.append(chosen_box)
        
    return bboxes_after_nms

def cellboxes_to_boxes(out, out_type="preds", S=7):
    converted_pred = convert_cellboxes(out, out_type).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long() # (N, S*S, 6)
    all_bboxes = []

    for batch_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[batch_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes
    
def convert_cellboxes(out, out_type="preds", S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """

    out = out.to("cpu")
    batch_size = out.shape[0]
    # labels (N, S, S, C)
    if out_type == "labels":
        best_boxes = out[..., 21:25] # (N, S, S, 4)
    # predictions (N, S * S * C+5*B)
    elif out_type == "preds":
        out = out.reshape(batch_size, 7, 7, 30)
        bboxes1 = out[..., 21:25]
        bboxes2 = out[..., 26:30]
        scores = torch.cat((out[..., 20].unsqueeze(0), out[..., 25].unsqueeze(0)), dim=0)
        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2 # (N, S, S, 4)
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)  # (N, S, S, 4)
    predicted_class = out[..., :20].argmax(-1).unsqueeze(-1)
    if out_type == "labels":
        best_confidence = out[..., 20].unsqueeze(-1)
        converted_bboxes = converted_bboxes * best_confidence
    elif out_type == 'preds':
        best_confidence = torch.max(out[..., 20], out[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def get_bboxes(loader, model, iou_threshold, cs_threshold, device="cuda"):
    """
    Return all ground truths and all prediction bounding boxes, possibly 
    zero according to cs_threshold level, for all training images
    """
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0 # image_idx accross all training images

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device) # (N, S, S, C+5)

        with torch.no_grad():
            predictions = model(x) # (N, S * S * C+5*B)

        batch_size = x.shape[0]
        # in each batch convert labels and predictions' cells to image ratio cordinates 
        true_bboxes = cellboxes_to_boxes(labels, out_type="labels")
        pred_bboxes = cellboxes_to_boxes(predictions, out_type="preds") # [[[c, cs, x, y, w, h] x S*S] x N]

        # for each image in the batch
        for idx in range(batch_size):
            nms_boxes = nms(pred_bboxes[idx], iou_threshold, cs_threshold)


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            # add [train_idx] in front of all nms_boxes & true_bboxes[idx]
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0, just keep cs = 1 boxes
                if box[1] > cs_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
def mAP(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20, plot=True):
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x, y, w, h],...]
    average_precisions = []
    epsilon = 1e-6
    
    for c in range(num_classes):
        detections = []
        ground_truths = [] # for particular class c
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
                
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # img 0 has 3 bboxes for class c
        # img 1 has 5 bboxes for class c
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) # just to keep track which bboxes are covered so far
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            
        # amount_boxes = {0:torch.tensor([0,0,0], 1:torch.tensor([0,0,0,0,0])}
        detections.sort(key=lambda x: x[2], reverse=True) # for bypassing overlapping bboxes with less confidence score
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            # get all the ground truth bboxes corresponding to that image for this detection
            ground_truths_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            
            best_iou = 0
            for idx, gt in enumerate(ground_truths_img):
                iou = IoU(torch.tensor(detection[3:]), torch.tensor(gt[3:]))
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
                
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        print(precisions)
        print(recalls)
        average_precisions.append(torch.trapz(precisions, recalls))
        
        if plot:
            plt.plot(recalls, precisions, 'o-', label='precision-recall curve')
            plt.fill_between(recalls, precisions, color='lightblue', alpha=0.3, label='Area')
            plt.xlabel('recalls')
            plt.ylabel('precisions')
            plt.title(f'precision-recall curve for class {c}')
            plt.legend()
            plt.grid(True)
            plt.show()
        
    return sum(average_precisions) / len(average_precisions), precisions, recalls

# test
# true_boxes = [
#     [0, 0, 1.0, 0.6, 0.2, 0.7, 0.4],  # img_number, class_idx, confidence_score, x, y, w, h
#     [0, 0, 1.0, 0.3, 0.4, 0.2, 0.3],
#     [1, 0, 1.0, 0.2, 0.3, 0.2, 0.4],
#     [1, 0, 1.0, 0.4, 0.5, 0.3, 0.2],
#     [2, 0, 1.0, 0.5, 0.2, 0.8, 0.4],
#     [2, 0, 1.0, 0.1, 0.4, 0.3, 0.5],
#     [2, 0, 1.0, 0.5, 0.6, 0.2, 0.3],
#     [3, 0, 1.0, 0.4, 0.6, 0.8, 0.2],
#     [3, 0, 1.0, 0.5, 0.4, 0.5, 0.3],
#     [4, 0, 1.0, 0.3, 0.2, 0.2, 0.7],
#     [4, 0, 1.0, 0.6, 0.1, 0.3, 0.6],
#     [4, 0, 1.0, 0.6, 0.6, 0.2, 0.3],
# ]

# pred_boxes = [
#     [0, 0, 0.86, 0.66, 0.22, 0.60, 0.45],  # img_number, class_idx, confidence_score, x, y, w, h
#     [0, 0, 0.94, 0.31, 0.41, 0.22, 0.32],
#     [1, 0, 0.97, 0.21, 0.32, 0.21, 0.41],
#     [1, 0, 0.91, 0.41, 0.52, 0.29, 0.28],
#     [2, 0, 0.92, 0.33, 0.22, 0.21, 0.31],
#     [2, 0, 0.79, 0.34, 0.39, 0.31, 0.62],
#     [2, 0, 0.95, 0.59, 0.69, 0.22, 0.23],
#     [3, 0, 0.92, 0.74, 0.34, 0.41, 0.82],
#     [3, 0, 0.67, 0.89, 0.23, 0.49, 0.71],
#     [4, 0, 0.90, 0.33, 0.25, 0.21, 0.52],
#     [4, 0, 0.88, 0.65, 0.42, 0.37, 0.22],
# ]

# a, p, r = mAP(pred_boxes, true_boxes)
# print(p)
# print(r)