from tqdm import tqdm
import torch
from utils import process_batch_forward, process_batch_reverse
import torch.nn.functional as F

# --- Training Loop (Adapted for nnU-Net style) ---
def train_loop(dataloader, model, loss_fn, optimizer, scheduler, accumulation_steps, device):
    """Performs one epoch of training resembling nnU-Net practices."""
    model.train()
    total_loss = 0.0
    processed_batches = 0 # Tracks effective batches (after accumulation)

    optimizer.zero_grad()

    # Determine total iterations for this epoch for tqdm progress bar
    total_iters_in_epoch = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=total_iters_in_epoch, desc="Training")
    for batch_idx, (X, y) in pbar:
        #### !!! CRITICAL FOR NNUNET STYLE !!! ####
        # Apply extensive Data Augmentation HERE
        # This is ideally done inside your Dataset __getitem__ or using
        # a Pytorch augmentation library (Albumentations, batchgenerators)
        # Examples: Random rotations, scaling, elastic deform, gamma, contrast...
        # X, y = your_augmentation_function(X, y)
        #### ------------------------------------ ####

        X, y = X.to(device), y.to(device)

        # Forward pass (No AMP used by default in nnU-Net)
        pred = model(X)
        loss = loss_fn(pred, y) # Combined Dice+CE loss

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Optimizer step after accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_iters_in_epoch:
            optimizer.step()
            scheduler.step() # Step the scheduler after optimizer step
            optimizer.zero_grad()

            # Log loss and update progress bar
            # Note: Logging unscaled loss from the *last* micro-batch in accumulation cycle
            total_loss += loss.item()
            processed_batches += 1
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

            # Optional memory check (keep if useful)
            if processed_batches == 1:
                try:
                    # print(f"Effective batch done. Memory allocated: {torch.cuda.memory_allocated(device)} bytes", flush=True)
                    pass
                except Exception:
                    pass

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    print(f"Training Avg loss (per effective batch): {avg_loss:>8f}")
    print(f"End of Epoch LR: {optimizer.param_groups[0]['lr']:>8f}")
    return avg_loss

# --- Evaluation Loop (Modified for aggregated IoU) ---
def eval_loop(dataloader, model, loss_fn, device, target_size=512):
    """
    Evaluation loop calculating loss, aggregated Dice, and aggregated IoU.

    Args:
        dataloader: yields batches of (list[Tensor(C,H,W)], list[Tensor(H,W)])
        model: the neural network model (on device)
        loss_fn: the combined loss function (e.g., DiceCELoss) used for training
        device: the torch device (cuda or cpu)
        target_size: the size the model expects for input
    """
    model.eval()
    num_images_processed = 0
    total_loss = 0.0
    num_classes = -1 # Will be determined from first prediction

    # --- Aggregation Containers (CPU tensors recommended) ---
    # For Dice (using the memory-efficient method's components)
    total_dice_intersect = None # Shape [C]
    total_dice_sum_pred = None  # Shape [C]
    total_dice_sum_gt = None    # Shape [C]

    # For IoU
    total_iou_intersection = None # Shape [C]
    total_iou_union = None      # Shape [C]
    # --------------------------------------------------------

    # Use ignore_index from the main loss function for consistency
    ignore_index = getattr(loss_fn, 'ignore_index', 3)
    smooth_eval = getattr(loss_fn.dice, 'smooth', 1e-5) if hasattr(loss_fn, 'dice') else 1e-5 # Match smooth

    with torch.no_grad():
        for X_batch_list, y_batch_list in tqdm(dataloader, desc="Eval"):
            # 1. Forward Transform Inputs
            try:
                 X_processed, meta_list = process_batch_forward(X_batch_list, target_size=target_size)
            except NameError: 
                raise ValueError("`process_batch_forward` not found.")

            X_processed = X_processed.to(device)

            # 2. Model Inference
            pred_processed = model(X_processed) # Logits [N, C, target, target]

            # --- Determine num_classes once ---
            if num_classes == -1:
                num_classes = pred_processed.shape[1]
                # Initialize aggregation tensors now that we know num_classes
                total_dice_intersect = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
                total_dice_sum_pred = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
                total_dice_sum_gt = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
                total_iou_intersection = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
                total_iou_union = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
            # -----------------------------------

            # 3. Reverse Transform Outputs
            try:
                 pred_original_list = process_batch_reverse(pred_processed, meta_list, interpolation='bilinear')
            except NameError: raise ValueError("`process_batch_reverse` not found.")

            # 4. Compute Loss & Accumulate Metrics per Image
            current_batch_size = len(y_batch_list)
            for i in range(current_batch_size):
                # --- Prepare single image prediction and label ---
                pred_single_logits = pred_original_list[i].to(device) # [C, H_orig, W_orig]
                label_single_orig = y_batch_list[i].to(device) # [H_orig, W_orig] or [1, H_orig, W_orig]

                pred_single_batched = pred_single_logits.unsqueeze(0) # [1, C, H, W]
                label_single_batched = label_single_orig.unsqueeze(0) # [1, H, W] or [1, 1, H, W]

                # Convert label to index map if needed [1, H, W]
                if label_single_batched.ndim == 4 and label_single_batched.shape[1] == 1:
                    label_single_idxmap = label_single_batched.squeeze(1) # [1, H, W]
                elif label_single_batched.ndim == 3:
                    label_single_idxmap = label_single_batched # Already [1, H, W]
                else:
                    raise ValueError(f"Unsupported label shape: {label_single_batched.shape}")

                # --- Calculate Loss ---
                loss = loss_fn(pred_single_batched, label_single_batched) # Use original batch dim label for loss
                total_loss += loss.item()

                # --- Get Hard Predictions ---
                pred_single_hard = torch.argmax(pred_single_logits, dim=0) # [H_orig, W_orig]

                # --- Calculate & Accumulate Dice Components ---
                # Re-calculate necessary components for Dice aggregation
                probs_single = F.softmax(pred_single_logits, dim=0) # [C, H, W]
                gt_single_long = label_single_idxmap.squeeze(0).long() # [H, W] Long type needed

                mask = None
                if ignore_index is not None:
                    mask = (gt_single_long != ignore_index) # [H,W]

                gt_onehot = F.one_hot(gt_single_long, num_classes=num_classes).permute(2, 0, 1).float() # [C, H, W]

                if mask is not None:
                    gt_onehot = gt_onehot * mask.unsqueeze(0) # Apply mask [C, H, W]
                    probs_single_masked = probs_single * mask.unsqueeze(0)
                else:
                    probs_single_masked = probs_single

                # Sum over spatial H, W -> Shape [C]
                intersect_dice = (probs_single_masked * gt_onehot).sum(dim=(1, 2))
                sum_pred_dice = probs_single_masked.sum(dim=(1, 2))
                sum_gt_dice = gt_onehot.sum(dim=(1, 2)) # Use masked gt

                # Accumulate on CPU
                total_dice_intersect += intersect_dice.cpu().to(torch.float64)
                total_dice_sum_pred += sum_pred_dice.cpu().to(torch.float64)
                total_dice_sum_gt += sum_gt_dice.cpu().to(torch.float64)

                # --- Calculate & Accumulate IoU Components ---
                pred_hard_onehot = F.one_hot(pred_single_hard, num_classes=num_classes).permute(2, 0, 1).bool() # [C, H, W]
                gt_onehot_bool = gt_onehot.bool() # Use the already created (and potentially masked) one-hot GT

                if mask is not None:
                    pred_hard_onehot_masked = pred_hard_onehot & mask.unsqueeze(0) # Apply ignore mask
                else:
                    pred_hard_onehot_masked = pred_hard_onehot

                # Calculate intersection and union per class using boolean logic
                intersection_iou = (pred_hard_onehot_masked & gt_onehot_bool).sum(dim=(1, 2)) # [C]
                union_iou = (pred_hard_onehot_masked | gt_onehot_bool).sum(dim=(1, 2)) # [C]

                # Accumulate on CPU
                total_iou_intersection += intersection_iou.cpu().to(torch.float64)
                total_iou_union += union_iou.cpu().to(torch.float64)
                # -----------------------------------------------

                num_images_processed += 1

    # --- Calculate Final Average Metrics ---
    if num_images_processed == 0: # Handle empty dataloader case
        print("Evaluation dataloader was empty.")
        return 0.0, 0.0, 0.0 # Loss, Dice, IoU

    avg_loss = total_loss / num_images_processed

    # --- Final Aggregated Dice Calculation ---
    # Using Micro-average: (Sum of numerators) / (Sum of denominators)
    dice_numerator = 2. * total_dice_intersect + smooth_eval
    dice_denominator = total_dice_sum_pred + total_dice_sum_gt + smooth_eval
    # Create mask for valid classes (excluding ignore_index)
    valid_class_mask_dice = torch.ones(num_classes, dtype=torch.bool)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        valid_class_mask_dice[ignore_index] = False

    # Calculate micro average score over valid classes
    avg_dice_micro = 0.0
    if valid_class_mask_dice.sum() > 0:
        avg_dice_micro = (dice_numerator[valid_class_mask_dice].sum() /
                         torch.clip(dice_denominator[valid_class_mask_dice].sum(), 1e-8)).item()

    # --- Optional: Macro Average Dice ---
    per_class_dice = dice_numerator / torch.clip(dice_denominator, 1e-8)
    avg_dice_macro = 0.0
    if valid_class_mask_dice.sum() > 0:
        avg_dice_macro = per_class_dice[valid_class_mask_dice].mean().item()


    # --- Final Aggregated IoU Calculation ---
    # Create mask for valid classes (excluding ignore_index) for IoU
    valid_class_mask_iou = torch.ones(num_classes, dtype=torch.bool)
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        valid_class_mask_iou[ignore_index] = False

    # Calculate per-class IoU using aggregated counts
    # Add epsilon to denominator for stability
    epsilon = 1e-8
    per_class_iou = total_iou_intersection / (total_iou_union + epsilon)

    # Calculate Mean IoU (mIoU) over valid classes
    mean_iou = 0.0
    if valid_class_mask_iou.sum() > 0:
        mean_iou = per_class_iou[valid_class_mask_iou].mean().item()


    print(f"\n--- Evaluation Complete ---")
    print(f"  Images Processed: {num_images_processed}")
    print(f"  Average Loss (Original Size): {avg_loss:>8f}")
    print(f"  Micro Avg Dice Score ({valid_class_mask_dice.sum().item()} classes): {avg_dice_micro:>8f}")
    print(f"  Macro Avg Dice Score ({valid_class_mask_dice.sum().item()} classes): {avg_dice_macro:>8f}")
    print(f"  Mean IoU (mIoU) ({valid_class_mask_iou.sum().item()} classes): {mean_iou:>8f}")
    print(f"  --- Per-Class IoU ---")
    for c in range(num_classes):
        if valid_class_mask_iou[c]: # Only print for valid classes
            print(f"    Class {c}: {per_class_iou[c].item():>8f}")
        else:
            print(f"    Class {c}: Ignored")
    print("-" * 25)

    # Return relevant metrics for model saving (e.g., micro dice and mIoU)
    return avg_loss, avg_dice_micro, mean_iou