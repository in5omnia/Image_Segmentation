import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from utils.utils import process_batch_forward, process_batch_reverse
from utils.MetricsHistory import MetricsHistory
from torchvision.transforms import InterpolationMode

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train_loop(dataloader, model, loss_fn, optimizer, accumulation_steps, device, scheduler=None, target_size = None):
    """
    Performs one epoch of training.
    Args:
        dataloader: DataLoader for training data.
        model: The neural network model.
        loss_fn: The loss function.
        optimizer: The optimizer.
        accumulation_steps: Number of steps to accumulate gradients before updating.
        device: The device to train on (CPU or GPU).
        scheduler: Learning rate scheduler (optional).
        target_size: Target size for image resizing (optional).
    """
    model.train()
    total_loss = 0.0
    processed_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for batch_idx, (X, y) in pbar:

        if target_size is not None:
            # Resize images and labels if target_size is specified
            X, _ = process_batch_forward(X, target_size=target_size)
            y, _ = process_batch_forward(y, target_size=target_size, interpolation=InterpolationMode.NEAREST)
        
        X, y = X.to(device), y.to(device).long()
        pred = model(X)
        loss = loss_fn(pred, y.squeeze(1))

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            processed_batches += 1
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    print(f"Training Avg loss (per effective batch): {avg_loss:>8f}")
    return avg_loss


def eval_loop(dataloader, model, loss_fn, device, target_size, agg):
    """
    Evaluation loop calculating loss, aggregated Dice, aggregated Acc, and aggregated IoU.

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
    num_classes = 4
    agg.reset()
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Eval"):

            X, meta_list = process_batch_forward(X, target_size=target_size)
            X = X.to(device)
            preds = model(X) # Logits [N, C, target, target]

            preds = process_batch_reverse(preds, meta_list, interpolation='bilinear')

            for pred, label in zip(preds, y):
                pred = pred.to(device) # (C,H,W)
                label = label.to(device).long() # (H,W)

                loss = loss_fn(pred.unsqueeze(0), label.unsqueeze(0).squeeze(1)) # Add batch dimension
                total_loss += loss.item()
                agg.accumulate(pred, label)
                
                num_images_processed += 1

    avg_loss = total_loss / num_images_processed

    mean_dice, mean_iou, mean_acc = agg.compute_epoch_metrics()
    per_class_iou = agg.get_last_per_class_iou()
    ignore_index = agg.get_ignore_index()

    print(f"\n--- Evaluation Complete ---")
    print(f"  Images Processed: {num_images_processed}")
    print(f"  Average Loss (Original Size): {avg_loss:>8f}")
    print(f"  Ignored Class : {ignore_index}")
    print(f"  Macro Avg Acc score: {mean_acc:>8f}")
    print(f"  Macro Avg Dice Score: {mean_dice:>8f}")
    print(f"  Mean IoU (mIoU): {mean_iou:>8f}")
    print(f"  --- Per-Class IoU ---")
    for c in range(num_classes):
        print(f"    Class {c}: {per_class_iou[c].item():>8f}")
    print("-" * 25)

    return avg_loss, mean_dice, mean_iou
  
def trainReconstruction(dataloader, model, loss_fn, optimizer, accumulation_steps):
    """
    Trains a reconstruction model.
    Args:
        dataloader: DataLoader for the training data.
        model: The reconstruction model.
        loss_fn: The loss function.
        optimizer: The optimizer.
        accumulation_steps: Number of steps to accumulate gradients.
    """
    losses = []
    model.train()
    for batch_idx, (X, _) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Training")):
        X = X.to(device)
        # Compute prediction
        pred = model(X)

        # Compute loss
        loss = loss_fn(pred, X)
        losses.append(loss.item())
        scaled_loss = loss / accumulation_steps

        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

    return np.mean(losses)

def train_loop_prompt(dataloader, model, loss_fn, optimizer, accumulation_steps, device, scheduler=None, target_size = None):
    """
    Performs one epoch of training for a prompt-based model.
    Args:
        dataloader: DataLoader for training data.
        model: The prompt-based model.
        loss_fn: The loss function.
        optimizer: The optimizer.
        accumulation_steps: Number of steps to accumulate gradients.
        device: The device to train on (CPU or GPU).
        scheduler: Learning rate scheduler (optional).
        target_size: Target size for image resizing (optional).
    """
    model.train()
    total_loss = 0.0
    processed_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for batch_idx, (X, p, y) in pbar:

        if target_size is not None:
            X, _ = process_batch_forward(X, target_size=target_size)
            p, _ = process_batch_forward(p, target_size=target_size)
            y, _ = process_batch_forward(y, target_size=target_size, interpolation=InterpolationMode.NEAREST)

        X, p, y = X.to(device), p.to(device), y.to(device).long()
        pred = model(X, p)
        loss = loss_fn(pred, y.squeeze(1))

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            processed_batches += 1
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0
    print(f"Training Avg loss (per effective batch): {avg_loss:>8f}")
    return avg_loss

  
def evalReconstruction(dataloader, model, loss_fn, target_size, interpolation = 'bilinear'):
    """
    Evaluates a reconstruction model.
    Args:
        dataloader: DataLoader for the evaluation data.
        model: The reconstruction model.
        loss_fn: The loss function.
        target_size: Target size for image resizing.
        interpolation: Interpolation mode for resizing.
    """
    model.eval()
    num_batches = len(dataloader)
    total_loss = 0.0
    losses = []
    with torch.no_grad():
        for batch, (original_X, _) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Evaluation")):
            resized_X, meta_list = process_batch_forward(original_X, target_size=target_size)   # resize X for network
            resized_X = resized_X.to(device)

            # Compute prediction
            pred = model(resized_X)

            pred = process_batch_reverse(pred, meta_list, interpolation=interpolation)

            for p, label in zip(pred, original_X):
                # Move individual prediction and label to the device
                p = p.to(device).unsqueeze(0)  # Add batch dimension
                label = label.to(device).unsqueeze(0)  # Add batch dimension and ensure type is long

                if label.shape[1] == 4 and label.ndim == 4:
                    label = label[:, :3, :, :] # RGBA to RGB

                loss = loss_fn(p, label.squeeze(1))
                total_loss += loss.item()
                # Loss list
                losses.append(loss.item())

    return total_loss / num_batches, np.mean(losses)


def eval_loop_prompt(dataloader, model, loss_fn, device, target_size, agg):
    """
    Evaluation loop calculating loss, aggregated Dice, and aggregated IoU for a prompt-based model.

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
    num_classes = 4
    
    with torch.no_grad():
        for X, p, y in tqdm(dataloader, desc="Eval"):

            X, meta_list = process_batch_forward(X, target_size=target_size)
            p, _ = process_batch_forward(p, target_size=target_size)
            X, p = X.to(device), p.to(device)
            preds = model(X, p) # Logits [N, C, target, target]

            preds = process_batch_reverse(preds, meta_list, interpolation='bilinear')

            for pred, label in zip(preds, y):
                pred = pred.to(device) # (C,H,W)
                label = label.to(device).long() # (H,W)

                loss = loss_fn(pred.unsqueeze(0), label.unsqueeze(0).squeeze(1)) # Add batch dimension
                total_loss += loss.item()
                agg.accumulate(pred, label)
                
                num_images_processed += 1

    avg_loss = total_loss / num_images_processed

    mean_dice, mean_iou, mean_acc = agg.compute_epoch_metrics()
    per_class_iou = agg.get_last_per_class_iou()
    ignore_index = agg.get_ignore_index()

    print(f"\n--- Evaluation Complete ---")
    print(f"  Images Processed: {num_images_processed}")
    print(f"  Average Loss (Original Size): {avg_loss:>8f}")
    print(f"  Ignored Class : {ignore_index}")
    print(f"  Macro Avg Acc score: {mean_acc:>8f}")
    print(f"  Macro Avg Dice Score: {mean_dice:>8f}")
    print(f"  Mean IoU (mIoU): {mean_iou:>8f}")
    print(f"  --- Per-Class IoU ---")
    for c in range(num_classes):
        print(f"    Class {c}: {per_class_iou[c].item():>8f}")
    print("-" * 25)

    return avg_loss, mean_dice, mean_iou


def start_prompt(
        model_save_dir: str,
        model_save_name: str,
        model: nn.Module,
        optimizer: torch.optim,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        accumulation_steps: int,
        device: torch.device,
        train_loss_fn: nn.Module,
        val_loss_fn: nn.Module,
        target_size: int,
        scheduler: torch.optim.lr_scheduler = None,
        agg: MetricsHistory = None,
        load: bool = True,
        save: bool = True,
        num_classes: int = 4,
        ignore_index: int = 3,
        epochs: int = 100,
):
    """
    Starts the training pipeline for a prompt-based model.
    Args:
        model_save_dir: Directory to save the model.
        model_save_name: Name of the model file.
        model: The prompt-based model.
        optimizer: The optimizer.
        train_dataloader: DataLoader for the training data.
        val_dataloader: DataLoader for the validation data.
        accumulation_steps: Number of steps to accumulate gradients.
        device: The device to train on (CPU or GPU).
        train_loss_fn: The loss function for training.
        val_loss_fn: The loss function for validation.
        target_size: Target size for image resizing.
        scheduler: Learning rate scheduler (optional).
        agg: MetricsHistory object (optional).
        load: Whether to load a checkpoint (default: True).
        save: Whether to save the model (default: True).
        num_classes: Number of classes (default: 4).
        ignore_index: Index to ignore in the loss calculation (default: 3).
        epochs: Number of training epochs (default: 100).
    """
    start_epoch = 0
    best_dev_dice = -np.inf
    best_dev_miou = -np.inf
    best_dev_loss = np.inf

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(f"{model_save_dir}/metrics", exist_ok=True)
    if load and os.path.isfile(f"{model_save_dir}/{model_save_name}"):
        print(f"Loading checkpoint from: {model_save_dir}/{model_save_name}")

        checkpoint = torch.load(f"{model_save_dir}/{model_save_name}", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print(" -> Model state loaded.")

        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(" -> Optimizer state loaded.")
        except Exception as e:
            print(f" -> Warning: Could not load optimizer state: {e}. Optimizer will start from scratch.")

        # Load scheduler state
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(" -> Scheduler state loaded.")
        except Exception as e:
            print(f" -> Warning: Could not load scheduler state: {e}. Scheduler will start from scratch.")

        # Load Metrics History
        try:
            agg = checkpoint.get("history")
            agg.to(device)
            print(" -> Metrics History loaded.")
        except Exception as e:
            print(f" -> No metric history saved")
            agg = MetricsHistory(num_classes, ignore_index)

        # Load training metadata
        start_epoch = checkpoint.get("epoch", 0)
        best_dev_dice = checkpoint.get("best_dev_dice", -np.inf)
        best_dev_miou = checkpoint.get("best_dev_miou", -np.inf)
        best_dev_loss = checkpoint.get("best_dev_loss", np.inf)

        print(f" -> Resuming training from epoch {start_epoch + 1}")
        print(f" -> Loaded best metrics: Dice={best_dev_dice:.6f}, mIoU={best_dev_miou:.6f}, Loss={best_dev_loss:.6f}")
        loaded_notes = checkpoint.get("notes", "N/A")
        print(f" -> Notes from checkpoint: {loaded_notes}")

    else:
        print(f"Checkpoint file not found at {model_save_dir}/{model_save_name}. Starting training from scratch.")

    # --- Training and Evaluation Loop ---
    print("\nStarting Training...")
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train_loss = train_loop_prompt(train_dataloader, model, train_loss_fn, optimizer, accumulation_steps, device, scheduler, target_size)
        val_loss, val_dice, val_miou = eval_loop_prompt(val_dataloader, model, val_loss_fn, device, target_size, agg)

        if save:
            metrics = {
                "epoch": t + 1,
                "history": agg
            }
            torch.save(metrics, f"{model_save_dir}/metrics/{model_save_name}")

        if val_miou > best_dev_miou:
            best_dev_dice = val_dice
            best_dev_miou = val_miou
            best_dev_loss = val_loss
            
            if save and scheduler:
                print(f"Validation IoU score improved ({best_dev_miou:.6f}). Saving model...")
                
                checkpoint = {
                    "epoch": t + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dev_dice": best_dev_dice,
                    "best_dev_miou": best_dev_miou,
                    "best_dev_loss": best_dev_loss,
                    "history": agg,
                    "notes": f"Model saved based on best Micro Dice. Ignored index for metric: {ignore_index}"
                }
                torch.save(checkpoint, f"{model_save_dir}/{model_save_name}")
            elif save:
                print(f"Validation IoU score improved ({best_dev_miou:.6f}). Saving model...")
                
                checkpoint = {
                    "epoch": t + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dev_dice": best_dev_dice,
                    "best_dev_miou": best_dev_miou,
                    "best_dev_loss": best_dev_loss,
                    "history": agg,
                    "notes": f"Model saved based on best Micro Dice. Ignored index for metric: {ignore_index}"
                }
                torch.save(checkpoint, f"{model_save_dir}/{model_save_name}")
        else:
            print(f"Validation IoU score did not improve from {best_dev_miou:.6f}")


    print("\n--- Training Finished! ---")
    print(f"Best validation IoU score achieved: {best_dev_miou:.6f}")
    print(f"Corresponding validation dice: {best_dev_dice:.6f}")
    print(f"Corresponding validation loss: {best_dev_loss:.6f}")
    print(f"Best model saved to: {os.path.join(model_save_dir, model_save_name)}")


def start(
        model_save_dir: str,
        model_save_name: str,
        model: nn.Module,
        optimizer: torch.optim,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        accumulation_steps: int,
        device: torch.device,
        train_loss_fn: nn.Module,
        val_loss_fn: nn.Module,
        target_size: int,
        scheduler: torch.optim.lr_scheduler = None,
        agg: MetricsHistory = None,
        load: bool = True,
        save: bool = True,
        num_classes: int = 4,
        ignore_index: int = 3,
        epochs: int = 100,
):
    """
    Starts the training pipeline.
    Args:
        model_save_dir: Directory to save the model.
        model_save_name: Name of the model file.
        model: The model.
        optimizer: The optimizer.
        train_dataloader: DataLoader for the training data.
        val_dataloader: DataLoader for the validation data.
        accumulation_steps: Number of steps to accumulate gradients.
        device: The device to train on (CPU or GPU).
        train_loss_fn: The loss function for training.
        val_loss_fn: The loss function for validation.
        target_size: Target size for image resizing.
        scheduler: Learning rate scheduler (optional).
        agg: MetricsHistory object (optional).
        load: Whether to load a checkpoint (default: True).
        save: Whether to save the model (default: True).
        num_classes: Number of classes (default: 4).
        ignore_index: Index to ignore in the loss calculation (default: 3).
        epochs: Number of training epochs (default: 100).
    """
    start_epoch = 0
    best_dev_dice = -np.inf
    best_dev_miou = -np.inf
    best_dev_loss = np.inf

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(f"{model_save_dir}/metrics", exist_ok=True)
    if load and os.path.isfile(f"{model_save_dir}/{model_save_name}"):
        print(f"Loading checkpoint from: {model_save_dir}/{model_save_name}")

        # Load checkpoint
        checkpoint = torch.load(f"{model_save_dir}/{model_save_name}", map_location=device, weights_only=True)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print(" -> Model state loaded.")

        # Load optimizer state
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(" -> Optimizer state loaded.")
        except Exception as e:
            print(f" -> Warning: Could not load optimizer state: {e}. Optimizer will start from scratch.")

        # Load scheduler state
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(" -> Scheduler state loaded.")
        except Exception as e:
            print(f" -> Warning: Could not load scheduler state: {e}. Scheduler will start from scratch.")

        # Load Metrics History
        try:
            agg = checkpoint.get("history")
            agg.to(device)
            print(" -> Metrics History loaded.")
        except Exception as e:
            print(f" -> No metric history saved")
            agg = MetricsHistory(num_classes, ignore_index)

        # Load training metadata
        start_epoch = checkpoint.get("epoch", 0)
        best_dev_dice = checkpoint.get("best_dev_dice", -np.inf)
        best_dev_miou = checkpoint.get("best_dev_miou", -np.inf)
        best_dev_loss = checkpoint.get("best_dev_loss", np.inf)

        print(f" -> Resuming training from epoch {start_epoch + 1}")
        print(f" -> Loaded best metrics: Dice={best_dev_dice:.6f}, mIoU={best_dev_miou:.6f}, Loss={best_dev_loss:.6f}")
        loaded_notes = checkpoint.get("notes", "N/A")
        print(f" -> Notes from checkpoint: {loaded_notes}")

    else:
        print(f"Checkpoint file not found at {model_save_dir}/{model_save_name}. Starting training from scratch.")

    # --- Training and Evaluation Loop ---
    print("\nStarting Training...")
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        
        train_loss = train_loop(train_dataloader, model, train_loss_fn, optimizer, accumulation_steps, device, scheduler, target_size)
        val_loss, val_dice, val_miou = eval_loop(val_dataloader, model, val_loss_fn, device, target_size, agg)

        if save:
            metrics = {
                "epoch": t + 1,
                "history": agg
            }
            torch.save(metrics, f"{model_save_dir}/metrics/{model_save_name}")

        if val_miou > best_dev_miou:
            best_dev_dice = val_dice
            best_dev_miou = val_miou
            best_dev_loss = val_loss
            
            if save and scheduler:
                print(f"Validation IoU score improved ({best_dev_miou:.6f}). Saving model...")
                
                checkpoint = {
                    "epoch": t + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_dev_dice": best_dev_dice,
                    "best_dev_miou": best_dev_miou,
                    "best_dev_loss": best_dev_loss,
                    # "history": agg,
                    "notes": f"Model saved based on best Micro Dice. Ignored index for metric: {ignore_index}"
                }
                torch.save(checkpoint, f"{model_save_dir}/{model_save_name}")
                # Save model weights only
                checkpoint = {
                    "epoch": t + 1,
                    "model_state_dict": model.state_dict(),
                }
                torch.save(checkpoint, f"{model_save_dir}/MO_{model_save_name}")
            elif save:
                print(f"Validation IoU score improved ({best_dev_miou:.6f}). Saving model...")
                
                checkpoint = {
                    "epoch": t + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dev_dice": best_dev_dice,
                    "best_dev_miou": best_dev_miou,
                    "best_dev_loss": best_dev_loss,
                    # "history": agg,
                    "notes": f"Model saved based on best Micro Dice. Ignored index for metric: {ignore_index}"
                }
                torch.save(checkpoint, f"{model_save_dir}/{model_save_name}")
                # Save model weights only
                checkpoint = {
                    "epoch": t + 1,
                    "model_state_dict": model.state_dict(),
                }
                torch.save(checkpoint, f"{model_save_dir}/MO_{model_save_name}")
        else:
            print(f"Validation IoU score did not improve from {best_dev_miou:.6f}")


    print("\n--- Training Finished! ---")
    print(f"Best validation IoU score achieved: {best_dev_miou:.6f}")
    print(f"Corresponding validation dice: {best_dev_dice:.6f}")
    print(f"Corresponding validation loss: {best_dev_loss:.6f}")
    print(f"Best model saved to: {os.path.join(model_save_dir, model_save_name)}")