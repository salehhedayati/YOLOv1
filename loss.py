import torch
import torch.nn as nn
from utils import IoU
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions = [p0, ..., p19, c, x1, y1, w1, h1, x2, y2, w2, h2]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        iou_b1 = IoU(predictions[..., 21:25], target[..., 21:25]) # (N, S, S, 1)
        iou_b2 = IoU(predictions[..., 26:30], target[..., 21:25]) # (N, S, S, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) # (2, N, S, S, 1)
        iou_maxes, best_box = torch.max(ious, dim=0) # (N, S, S, 1)
        exists_box = target[..., 20].unsqueeze(-1) # 1obj_i (N, S, S, 1)
        
        # =========================#
        # For Box Cordinates Loss  #
        # =========================#
        box_predictions = exists_box * (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        box_predictions_clone = box_predictions.clone() # to avoid in-place operation, it is a variables needed for gradient computation
        
        box_targets = exists_box * target[..., 21:25]
        box_predictions_clone[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # finally mse compute the squared difference between corresponding elements
        # (N, S, S, 4)
        box_loss = self.mse(
            box_predictions_clone,
            box_targets
            )
        
        # =========================#
        # For Object Loss          #
        # =========================#
        pred_box = best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        
        # (N, S, S, 1)
        object_loss = self.mse(
            exists_box * pred_box,
            exists_box * target[..., 20:21]
            )
        
        # =========================#
        # For No Object Loss       #
        # =========================#
        # (N, S, S, 1)
        no_obj_loss = self.mse(
            (1 - exists_box) * predictions[..., 20:21],
            (1 - exists_box) * target[..., 20:21]
            )
        no_obj_loss += self.mse(
            (1 - exists_box) * predictions[..., 25:26],
            (1 - exists_box) * target[..., 20:21]
            )
        
        # =========================#
        # For Class Loss           #
        # =========================#
        # (N, S, S, 20)
        class_loss = self.mse(
            exists_box * predictions[..., :20],
            exists_box * target[..., :20]
            )
        
        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_obj_loss + class_loss
        
        return loss
        