
# from drive.MyDrive.unet.unet import unet
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch
from torch import nn
from clipunet import ClipUNet

class DoubleConvReLU(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.doubleConvReLU = nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
            nn.Conv2d(dout, dout, kernel_size=3, padding=1),
            nn.BatchNorm2d(dout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.doubleConvReLU(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, din, dout):
        super().__init__()
        self.maxpool_doubleConv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvReLU(din, dout)
        )

    def forward(self, x):
        return self.maxpool_doubleConv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, din, dout):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(din, dout, kernel_size=2, stride=2)
        self.doubleConv = DoubleConvReLU(din, dout)

    def forward(self, x1, x2):
        x = torch.cat([x1, self.upsample(x2)], dim=1)
        return self.doubleConv(x)


class unet(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.scale = 1

        self.down1 = DoubleConvReLU(din, self.scale * 64)
        self.down2 = Down(self.scale * 64, self.scale * 128)
        self.down3 = Down(self.scale * 128, self.scale * 256)
        self.down4 = Down(self.scale * 256, self.scale * 512)
        self.down5 = Down(self.scale * 512, self.scale * 1024)

        self.up1 = Up(self.scale * 1024, self.scale * 512)
        self.up2 = Up(self.scale * 512, self.scale * 256)
        self.up3 = Up(self.scale * 256, self.scale * 128)
        self.up4 = Up(self.scale * 128, self.scale * 64)

        self.output = nn.Conv2d(self.scale * 64, dout, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        return self.output(x)
    

class PromptModel(nn.Module):
    def __init__(self, path):
        super().__init__()

        self.clip = ClipUNet()
        self.mask = unet(4, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        if path is not None:
            try:
                checkpoint = torch.load(path, weights_only=False, map_location=lambda storage, loc: storage) # Load to CPU initially
                self.clip.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)[:200]}")
                raise
            
        # for param in self.clip.parameters():
        #     param.requires_grad = False

    def forward(self, x, heatmap):
        clip_logit = self.clip(x)
        clip_prob = self.softmax(clip_logit)

        mask_logit = self.mask(torch.concat([x, heatmap], dim=1))
        mask_prob = self.sigmoid(mask_logit)

        final_probs = torch.empty_like(clip_prob)
        selected_prob = mask_prob * clip_prob

        final_probs[:, 1:4, :, :] = selected_prob[:, 0:3, :, :]
        final_probs[:, 0:1, :, :] = 1.0 - mask_prob
        final_probs[:, 1:2, :, :] += selected_prob[:, 3:4, :, :]

        # log_final_probs = torch.log(final_probs + 1e-9)

        return final_probs
    





#def eval_loop(dataloader, model, loss_fn, device, target_size, agg):
    """
    Evaluation loop calculating loss, aggregated Dice, and aggregated IoU.

    Args:
        dataloader: yields batches of (list[Tensor(C,H,W)], list[Tensor(H,W)])
        model: the neural network model (on device)
        loss_fn: the combined loss function (e.g., DiceCELoss) used for training
        device: the torch device (cuda or cpu)
        target_size: the size the model expects for input
    """
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

            # --- Define Class Names and Colormap for Legend ---
    class_names = ['deactivate', 'bg+bd', 'cat', 'dog'] # As requested
    num_classes = len(class_names)
    # Use a qualitative colormap suitable for distinct classes
    # 'tab10' provides 10 distinct colors. 'Paired' or 'Set1' are alternatives.
    cmap = plt.get_cmap('tab10', num_classes)
    # --- End Legend Definitions ---

    if 'X' not in locals() or X.nelement() == 0:
         print("No data processed in the epoch, skipping visualization.")
         avg_loss = 0
    else:
        # Prepare images for plotting (use the *last* batch)
        # Image 1: Input Image (convert to HWC for imshow)
        # Normalize or clip if necessary for display
        img1 = X[0].cpu().permute(1, 2, 0).numpy()
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-6) # Basic normalization for display

        # Image 2: Processed / Feature Map (assuming it's displayable like an image)
        # Handle single-channel vs multi-channel 'p'
        p_plot = p[0].cpu()
        if p_plot.shape[0] == 1: # Single channel (e.g., heatmap)
             img2 = p_plot.squeeze(0).numpy() # Keep grayscale H, W
             cmap_p = 'viridis' # Or 'gray'
        else: # Multi-channel (e.g., RGB-like features)
             img2 = p_plot.permute(1, 2, 0).numpy() # Convert to H, W, C
             img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-6) # Normalize
             cmap_p = None # imshow handles RGB directly

        # Image 3: Prediction (show the most likely class index per pixel)
        img3 = preds[0].argmax(dim=0).cpu().numpy() # Shape: (H, W)
        print("Unique preds (img3):", np.unique(img3))

        # Image 4: Ground Truth (should be class indices)
        # y[0] likely has shape (1, H, W), squeeze to get (H, W)
        img4 = y[0].squeeze(0).cpu().numpy() # Shape: (H, W)
        print("Unique GT labels (img4):", np.unique(img4)) # Check the actual labels present

        # --- Plotting ---
        fig, axes = plt.subplots(1, 4, figsize=(22, 5)) # Adjusted figsize for legend space

        # --- Plot Image 1 ---
        axes[0].imshow(img1)
        axes[0].set_title(f'Input Image (X[0])\nShape: {img1.shape}')
        axes[0].axis('off')

        # --- Plot Image 2 ---
        axes[1].imshow(img2, cmap=cmap_p) # Use specific cmap if grayscale
        axes[1].set_title(f'Processed (p[0])\nShape: {img2.shape}')
        axes[1].axis('off')

        # --- Plot Image 3 (Prediction) ---
        # Use the defined class colormap and ensure correct range
        im3 = axes[2].imshow(img3, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[2].set_title(f'Prediction (argmax)\nUnique IDs: {np.unique(img3)}')
        axes[2].axis('off')

        # --- Plot Image 4 (Ground Truth) ---
        # Use the same class colormap and range for comparison
        im4 = axes[3].imshow(img4, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[3].set_title(f'Ground Truth (y[0])\nUnique IDs: {np.unique(img4)}')
        axes[3].axis('off')

        # --- Create Legend ---
        legend_patches = []
        colors = [cmap(i) for i in range(num_classes)]
        for i in range(num_classes):
            patch = mpatches.Patch(color=colors[i], label=f'{i}: {class_names[i]}')
            legend_patches.append(patch)

        # Add legend outside the last plot (axes[3])
        axes[3].legend(handles=legend_patches,
                       bbox_to_anchor=(1.05, 1), # Position legend slightly to the right, top-aligned
                       loc='upper left',         # Anchor point of the legend box
                       borderaxespad=0.,        # No padding between anchor and legend
                       title="Classes")         # Optional title for the legend

        # Adjust layout to prevent titles overlapping and make space for legend
        # Option 1: tight_layout with rect (often works well)
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Leave space on the right (adjust 0.88 as needed)
        # Option 2: subplots_adjust (alternative)
        # fig.subplots_adjust(right=0.85, wspace=0.1) # Adjust right margin & width space

        plt.show()

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

"""