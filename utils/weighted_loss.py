import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class WeightedMemoryEfficientDiceLoss(nn.Module):
    """
    Calculates a memory-efficient Soft Dice loss, optionally with class weights.
    Args:
        apply_softmax (bool): Whether to apply softmax to the input logits. Defaults to True.
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to None.
        class_weights (torch.Tensor, optional): Weights for each class. Defaults to None.
        smooth (float): Smoothing factor to prevent division by zero. Defaults to 1e-5.
    """
    def __init__(self,
                 apply_softmax: bool = True,
                 ignore_index: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 smooth: float = 1e-5):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.ignore_index = ignore_index
        self.smooth = smooth

        if class_weights is not None:
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

        with torch.no_grad():
            # Handle potential shape mismatches between predictions and targets
            if len(shp_y) != len(probs.shape):
                if len(shp_y) == len(probs.shape) - 1 and len(shp_y) >= 2 and shp_y == probs.shape[2:]:
                     y = y.unsqueeze(1)
                elif len(shp_y) == len(probs.shape) and shp_y[1] == 1: pass # ok
                else: raise ValueError(f"Shape mismatch: probs {probs.shape}, y {shp_y}")
            y_long = y.long()

            mask = None
            if probs.shape == y.shape:
                 y_onehot = y.float()
                 if mask is not None:
                      y_indices_for_mask = torch.argmax(y_onehot, dim=1, keepdim=True)
                      mask = (y_indices_for_mask != self.ignore_index)
                      y_onehot = y_onehot * mask
            else:
                y_onehot = torch.zeros_like(probs, device=probs.device)
                y_onehot.scatter_(1, y_long, 1)
                if mask is not None: y_onehot = y_onehot * mask

            sum_gt = y_onehot.sum(dim=(2, 3))

        if mask is not None:
             probs = probs * mask

        intersect_persample = (probs * y_onehot).sum(dim=(2, 3))
        sum_pred_persample = probs.sum(dim=(2, 3))
        sum_gt_persample = sum_gt

        # Aggregate across batch
        intersect = intersect_persample.sum(0)
        sum_pred = sum_pred_persample.sum(0)
        sum_gt = sum_gt_persample.sum(0)

        # Dice
        denominator = sum_pred + sum_gt
        dc = (2. * intersect + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        valid_classes_mask = torch.ones_like(dc, dtype=torch.bool)
        if self.ignore_index is not None and 0 <= self.ignore_index < num_classes:
            valid_classes_mask[self.ignore_index] = False

        dc_final = torch.tensor(0.0, device=dc.device)

        dc_valid = dc[valid_classes_mask]

        if self.class_weights is not None:
            # Use weighted average for valid classes
            weights = self.class_weights.to(dc_valid.device)
            weights_valid = weights[valid_classes_mask]
            
            weighted_sum = (dc_valid * weights_valid).sum()
            weight_sum = weights_valid.sum()
            dc_final = weighted_sum / weight_sum.clamp(min=1e-8)
        else:
            dc_final = dc_valid.mean()

        return -dc_final # Return negative Dice score as loss



class WeightedDiceCELoss(nn.Module):
    """
    Combines WeightedMemoryEfficientDiceLoss and Cross Entropy Loss with class_weights support.
    Args:
        dice_weight (float): Weight for the Dice loss. Defaults to 1.0.
        ce_weight (float): Weight for the Cross Entropy loss. Defaults to 1.0.
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to None.
        class_weights (torch.Tensor, optional): Weights for each class. Defaults to None.
        smooth_dice (float): Smoothing factor for Dice loss. Defaults to 1e-5.
        ce_kwargs (dict): Keyword arguments for CrossEntropyLoss. Defaults to {}.
    """
    def __init__(self,
                 dice_weight: float = 1.0,
                 ce_weight: float = 1.0,
                 ignore_index: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None, 
                 smooth_dice: float = 1e-5,
                 ce_kwargs={}):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index

        self.dice = WeightedMemoryEfficientDiceLoss(
            apply_softmax=True,
            ignore_index=ignore_index,
            class_weights=class_weights,
            smooth=smooth_dice
        )

        ce_final_kwargs = ce_kwargs.copy()
        if ignore_index is not None:
            ce_final_kwargs['ignore_index'] = ignore_index
        if class_weights is not None:
            ce_final_kwargs['weight'] = class_weights

        self.cross_entropy = nn.CrossEntropyLoss(**ce_final_kwargs)

    def forward(self, outputs, targets):
        if targets.ndim == 3:
            targets_dice = targets.unsqueeze(1).long() # Assuming [N, H, W]
        elif targets.ndim == 4 and targets.shape[1] == 1:
            targets_dice = targets.long()
        else:
            if targets.ndim == outputs.ndim and targets.shape[1] != 1:
                raise ValueError(f"Target shape {targets.shape} has multiple channels but expected class indices [N, H, W] or [N, 1, H, W] for CE.")
            else:
                raise ValueError(f"Unsupported target shape {targets.shape} for CE. Expected [N, H, W] or [N, 1, H, W].")

        dice_loss = self.dice(outputs, targets_dice)

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets_ce = targets.squeeze(1).long()
        elif targets.ndim == 3:
            targets_ce = targets.long()
        else:
            if targets.ndim == outputs.ndim and targets.shape[1] != 1:
                raise ValueError(f"Target shape {targets.shape} has multiple channels but expected class indices [N, H, W] or [N, 1, H, W] for CE.")
            else:
                raise ValueError(f"Unsupported target shape {targets.shape} for CE. Expected [N, H, W] or [N, 1, H, W].")

        ce_loss = self.cross_entropy(outputs, targets_ce)

        combined_loss = (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)
        return combined_loss



class WeightedMemoryEfficientDiceLossPrompt(nn.Module):
    """
    Calculates a memory-efficient Dice loss, optionally with class weights and a non-linearity.
    Args:
        dice_nonlin (Callable, optional): Non-linearity function to apply to the predictions. Defaults to None.
        apply_softmax (bool): Whether to apply softmax to the input logits. Defaults to True.
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to None.
        class_weights (torch.Tensor, optional): Weights for each class. Defaults to None.
        smooth (float): Smoothing factor to prevent division by zero. Defaults to 1e-5.
    """
    def __init__(self,
                 dice_nonlin: Callable = None,
                 apply_softmax: bool = True,
                 ignore_index: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None, # New parameter
                 smooth: float = 1e-5):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.dice_nonlin = dice_nonlin

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
            
        if self.dice_nonlin is not None:
            probs = self.dice_nonlin(probs)
            
        with torch.no_grad():
            if len(shp_y) != len(probs.shape):
                if len(shp_y) == len(probs.shape) - 1 and len(shp_y) >= 2 and shp_y == probs.shape[2:]:
                     y = y.unsqueeze(1)
                elif len(shp_y) == len(probs.shape) and shp_y[1] == 1: pass # ok
                else: raise ValueError(f"Shape mismatch: probs {probs.shape}, y {shp_y}")
            y_long = y.long()

            mask = None
            if probs.shape == y.shape:
                 y_onehot = y.float()
                 if mask is not None:
                      y_indices_for_mask = torch.argmax(y_onehot, dim=1, keepdim=True)
                      mask = (y_indices_for_mask != self.ignore_index)
                      y_onehot = y_onehot * mask
            else:
                y_onehot = torch.zeros_like(probs, device=probs.device)
                y_onehot.scatter_(1, y_long, 1)
                if mask is not None: y_onehot = y_onehot * mask

            sum_gt = y_onehot.sum(dim=(2, 3))

        if mask is not None:
             probs = probs * mask

        intersect_persample = (probs * y_onehot).sum(dim=(2, 3))
        sum_pred_persample = probs.sum(dim=(2, 3))
        sum_gt_persample = sum_gt

        # Aggregate across batch
        intersect = intersect_persample.sum(0)
        sum_pred = sum_pred_persample.sum(0)
        sum_gt = sum_gt_persample.sum(0)

        # Dice
        denominator = sum_pred + sum_gt
        dc = (2. * intersect + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        valid_classes_mask = torch.ones_like(dc, dtype=torch.bool)
        if self.ignore_index is not None and 0 <= self.ignore_index < num_classes:
            valid_classes_mask[self.ignore_index] = False

        dc_final = torch.tensor(0.0, device=dc.device) 
        dc_valid = dc[valid_classes_mask]
        if self.class_weights is not None:
            weights = self.class_weights.to(dc_valid.device)
            weights_valid = weights[valid_classes_mask]
            weighted_sum = (dc_valid * weights_valid).sum()
            weight_sum = weights_valid.sum()
            dc_final = weighted_sum / weight_sum.clamp(min=1e-8)
        else:
            dc_final = dc_valid.mean()

        return -dc_final # Return negative Dice score as loss


class WeightedDiceNLLLoss(nn.Module):
    """
    Combines WeightedMemoryEfficientDiceLossPrompt and Cross Entropy Loss with class_weights support.
    Args:
        dice_weight (float): Weight for the Dice loss. Defaults to 1.0.
        nll_weight (float): Weight for the NLL loss. Defaults to 1.0.
        ignore_index (int, optional): Index to ignore in the loss calculation. Defaults to None.
        class_weights (torch.Tensor, optional): Weights for each class. Defaults to None.
        smooth_dice (float): Smoothing factor for Dice loss. Defaults to 1e-5.
        apply_softmax (bool): Whether to apply softmax to the input logits. Defaults to True.
        dice_nonlin (Callable, optional): Non-linearity function to apply to the predictions for Dice loss. Defaults to None.
        nll_nonlin (Callable, optional): Non-linearity function to apply to the predictions for NLL loss. Defaults to None.
        nll_kwargs (dict): Keyword arguments for NLLLoss. Defaults to {}.
    """
    def __init__(self,
                 dice_weight: float = 1.0,
                 nll_weight: float = 1.0,
                 ignore_index: Optional[int] = None,
                 class_weights: Optional[torch.Tensor] = None,
                 smooth_dice: float = 1e-5,
                 apply_softmax: bool = True,
                 dice_nonlin: Callable = None,
                 nll_nonlin: Callable = None,
                 nll_kwargs={}):
        super().__init__()
        self.dice_weight = dice_weight
        self.nll_weight = nll_weight
        self.ignore_index = ignore_index
        self.dice_nonlin = dice_nonlin
        self.nll_nonlin = nll_nonlin

        self.dice = WeightedMemoryEfficientDiceLossPrompt(
            apply_softmax=apply_softmax,
            ignore_index=ignore_index,
            class_weights=class_weights,
            smooth=smooth_dice
        )

        nll_final_kwargs = nll_kwargs.copy()
        if ignore_index is not None:
            nll_final_kwargs['ignore_index'] = ignore_index
        if class_weights is not None:
            nll_final_kwargs['weight'] = class_weights

        self.nll = nn.NLLLoss(**nll_final_kwargs)

    def forward(self, outputs, targets):
        if targets.ndim == 3:
            targets_dice = targets.unsqueeze(1).long()
        elif targets.ndim == 4 and targets.shape[1] == 1:
            targets_dice = targets.long()
        else:
            if targets.ndim == outputs.ndim and targets.shape[1] != 1:
                raise ValueError(f"Target shape {targets.shape} has multiple channels but expected class indices [N, H, W] or [N, 1, H, W] for CE.")
            else:
                raise ValueError(f"Unsupported target shape {targets.shape} for CE. Expected [N, H, W] or [N, 1, H, W].")


        dice_loss = self.dice(outputs, targets_dice)

        if targets.ndim == 4 and targets.shape[1] == 1:
             targets_nll = targets.squeeze(1).long()
        elif targets.ndim == 3:
             targets_nll = targets.long()
        else:
            if targets.ndim == outputs.ndim and targets.shape[1] != 1:
                raise ValueError(f"Target shape {targets.shape} has multiple channels but expected class indices [N, H, W] or [N, 1, H, W] for CE.")
            else:
                raise ValueError(f"Unsupported target shape {targets.shape} for CE. Expected [N, H, W] or [N, 1, H, W].")
             
        if self.nll_nonlin is not None:
            outputs = self.nll_nonlin(outputs)
        nll_loss = self.nll(outputs, targets_nll)
        
        combined_loss = (self.dice_weight * dice_loss) + (self.nll_weight * nll_loss)
        return combined_loss