import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for softmax in Dice Loss
from typing import Optional

# --- Modified Dice Loss with Class Weights ---
class WeightedMemoryEfficientDiceLoss(nn.Module):
    """ Version using ignore_index and supporting class weights """
    def __init__(self,
                 apply_softmax: bool = True,
                 ignore_index: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None, # New parameter
                 smooth: float = 1e-5):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.ignore_index = ignore_index
        self.smooth = smooth

        # Store class weights, ensuring they are a Tensor if provided
        if class_weights is not None:
            assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch.Tensor"
            self.class_weights = class_weights
        else:
            self.class_weights = None


    def forward(self, x, y):
        num_classes = x.shape[1]
        shp_y = y.shape
        if self.apply_softmax:
            probs = F.softmax(x, dim=1)
        else:
            probs = x

        # --- One-Hot Encoding and Masking ---
        with torch.no_grad():
            # Shape adjustments (same as before)
            if len(shp_y) != len(probs.shape):
                if len(shp_y) == len(probs.shape) - 1 and len(shp_y) >= 2 and shp_y == probs.shape[2:]:
                     y = y.unsqueeze(1)
                elif len(shp_y) == len(probs.shape) and shp_y[1] == 1: pass # ok
                else: raise ValueError(f"Shape mismatch: probs {probs.shape}, y {shp_y}")
            y_long = y.long()

            # Spatial mask based on ignore_index
            mask = None
            if self.ignore_index is not None:
                mask = (y_long != self.ignore_index)

            # Create one-hot ground truth (potentially masked)
            if probs.shape == y.shape: # Already one-hot
                 y_onehot = y.float()
                 if mask is not None:
                      y_indices_for_mask = torch.argmax(y_onehot, dim=1, keepdim=True)
                      mask = (y_indices_for_mask != self.ignore_index)
                      y_onehot = y_onehot * mask
            else: # Create from index map
                y_onehot = torch.zeros_like(probs, device=probs.device)
                y_onehot.scatter_(1, y_long, 1)
                if mask is not None: y_onehot = y_onehot * mask

            sum_gt = y_onehot.sum(dim=(2, 3)) # Pre-calculate GT sum needed later [N, C]
        # --- End One-Hot Encoding ---

        # Apply spatial mask to probabilities before summation
        if mask is not None:
             probs = probs * mask

        # Calculate intersection and prediction sum (still per-sample)
        intersect_persample = (probs * y_onehot).sum(dim=(2, 3)) # Shape [N, C]
        sum_pred_persample = probs.sum(dim=(2, 3))              # Shape [N, C]
        sum_gt_persample = sum_gt                               # Shape [N, C]

        # --- Aggregate across batch ---
        intersect = intersect_persample.sum(0) # Shape [C]
        sum_pred = sum_pred_persample.sum(0)   # Shape [C]
        sum_gt = sum_gt_persample.sum(0)     # Shape [C]

        # --- Calculate per-class Dice ---
        denominator = sum_pred + sum_gt
        dc = (2. * intersect + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8)) # Shape [C]

        # --- Average Dice Logic (Weighted or Unweighted) ---
        # Mask for valid (non-ignored) classes
        valid_classes_mask = torch.ones_like(dc, dtype=torch.bool)
        if self.ignore_index is not None and 0 <= self.ignore_index < num_classes:
            valid_classes_mask[self.ignore_index] = False

        dc_final = torch.tensor(0.0, device=dc.device) # Default loss if no valid classes

        if valid_classes_mask.sum() > 0: # Proceed only if there are valid classes
            dc_valid = dc[valid_classes_mask] # Dice scores for valid classes

            if self.class_weights is not None:
                # Use weighted average for valid classes
                weights = self.class_weights.to(dc_valid.device) # Ensure weights are on correct device
                weights_valid = weights[valid_classes_mask] # Select weights for valid classes
                # Calculate weighted mean: sum(value*weight) / sum(weight)
                weighted_sum = (dc_valid * weights_valid).sum()
                weight_sum = weights_valid.sum()
                dc_final = weighted_sum / weight_sum.clamp(min=1e-8) # Avoid division by zero
            else:
                # Use simple mean if no weights are provided
                dc_final = dc_valid.mean()

        return -dc_final # Return negative Dice score as loss


# --- Modified Combined Loss to use Weighted Dice and accept CE weights ---
class WeightedDiceCELoss(nn.Module): # Renamed for clarity
    """Combines WeightedMemoryEfficientDiceLoss and Cross Entropy Loss with class_weights support."""
    def __init__(self,
                 dice_weight: float = 1.0,
                 ce_weight: float = 1.0,
                 ignore_index: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None, # Pass weights here
                 smooth_dice: float = 1e-5, # Adjust smooth value as needed (e.g., 1.0 for training)
                 ce_kwargs={}):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index # Store ignore_index

        # Ensure class_weights is a Tensor if provided
        if class_weights is not None:
             assert isinstance(class_weights, torch.Tensor), "class_weights must be a torch.Tensor"

        # Instantiate the modified Dice loss, passing weights
        self.dice = WeightedMemoryEfficientDiceLoss(
            apply_softmax=True, # Dice loss usually works on probabilities
            ignore_index=ignore_index,
            class_weights=class_weights, # Pass weights to Dice component
            smooth=smooth_dice
        )

        # Prepare kwargs for standard CrossEntropyLoss
        ce_final_kwargs = ce_kwargs.copy()
        if ignore_index is not None:
            ce_final_kwargs['ignore_index'] = ignore_index
        if class_weights is not None:
            # Pass the weights tensor to CE's 'weight' parameter
            ce_final_kwargs['weight'] = class_weights
            # Note: CE will handle moving the weights tensor to the correct device internally

        self.cross_entropy = nn.CrossEntropyLoss(**ce_final_kwargs)

    def forward(self, outputs, targets):
        # outputs are expected to be logits [N, C, H, W]
        # targets are expected to be class indices [N, H, W] or [N, 1, H, W]

        # --- Dice Loss ---
        # WeightedMemoryEfficientDiceLoss handles softmax internally
        dice_loss = self.dice(outputs, targets)

        # --- Cross Entropy Loss ---
        # Prepare targets for CE
        if targets.ndim == 4 and targets.shape[1] == 1:
             targets_ce = targets.squeeze(1).long()
        elif targets.ndim == 3:
             targets_ce = targets.long() # Assuming [N, H, W]
        else:
             # Added more specific error message for common cases
             if targets.ndim == outputs.ndim and targets.shape[1] != 1:
                 raise ValueError(f"Target shape {targets.shape} has multiple channels but expected class indices [N, H, W] or [N, 1, H, W] for CE.")
             else:
                 raise ValueError(f"Unsupported target shape {targets.shape} for CE. Expected [N, H, W] or [N, 1, H, W].")

        # Weights are handled internally by nn.CrossEntropyLoss via its 'weight' parameter
        ce_loss = self.cross_entropy(outputs, targets_ce)
        
        # --- Combine ---
        combined_loss = (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)
        return combined_loss
