import torch
import torch.nn as nn
from utils import iou
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

        iou_b1 = iou(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = iou(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # 1obj_i
        
        # =========================#
        # For Box Cordinates Loss  #
        # =========================#
        box_predictions = exists_box * (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
            )
        
        # =========================#
        # For Object Loss          #
        # =========================#
        pred_box = best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        
        # (N*S*S*1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )
        
        # =========================#
        # For No Object Loss       #
        # =========================#
        # (N*S*S*1)
        no_obj_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21]),
            torch.flatten((1 - exists_box) * target[..., 20:21]),
        )
        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26]),
            torch.flatten((1 - exists_box) * target[..., 20:21]),
        )
        
        # =========================#
        # For Box Cordinates Loss  #
        # =========================#
        # (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )
        
        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_obj_loss + class_loss
        
        return loss
        
        
# # Example tensors
# pred_box = torch.randn(2, 2, 2, 1)
# target = torch.randn(2, 2, 2, 1)
# exists_box = torch.randn(2, 2, 2, 1)

# # Concatenate the tensors along the last dimension
# combined_tensor = torch.cat([tensor1.unsqueeze(0), tensor2.unsqueeze(0)], dim=0)

# # Find the maximum values and their corresponding indices along the last dimension
# max_values, max_indices = torch.max(combined_tensor, dim=0)

# print("Maximum values shape:", max_values.shape)  # Shape: (N, 7, 7)
# print("Maximum indices shape:", max_indices.shape)  # Shape: (N, 7, 7)

# target.shape = torch.rand((1, 7, 7, 25))