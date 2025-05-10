import torch
import torch.nn.functional as F

class MetricsHistory:
    """
    Accumulates TP, FP, FN, TN over an epoch for multi-class segmentation
    and computes Dice, IoU, and Accuracy metrics.
    """
    def __init__(self, num_classes: int, ignore_index: int = None, device: str = 'cpu'):
        """
        Initializes the MetricsHistory object.
        Args:
            num_classes (int): Number of classes including background.
            ignore_index (int, optional): Index of the class to ignore during metric calculation. Defaults to None.
            device (str): Device to perform initial calculations, results accumulated on CPU.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Initialize tensors to store the sums of TP, FP, FN, and TN.
        self.total_tp = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
        self.total_fp = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
        self.total_fn = torch.zeros(num_classes, dtype=torch.float64, device='cpu')
        self.total_tn = torch.zeros(num_classes, dtype=torch.float64, device='cpu')

        # History lists for epoch metrics
        self.epoch_mean_dice_history = []
        self.epoch_mean_iou_history = []
        self.epoch_mean_acc_history = []
        
        self.epoch_per_class_dice_history = []
        self.epoch_per_class_iou_history = []
        self.epoch_per_class_acc_history = []
        
        self.last_per_class_iou = None
        self.last_per_class_dice = None
        self.last_per_class_acc = None

        # Metric mask (calculated once)
        self.mask = torch.ones(num_classes, dtype=torch.bool)
        # Set mask element to False if ignore_index is specified.
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            self.mask[self.ignore_index] = False

    def reset(self):
        """
        Resets the accumulated TP, FP, FN, TN counts.
        """
        # Resets the accumulated statistics to zero at the start of each epoch.
        self.total_tp.zero_()
        self.total_fp.zero_()
        self.total_fn.zero_()
        self.total_tn.zero_()

    def accumulate(self, pred: torch.Tensor, label: torch.Tensor):
        """
        Accumulates statistics for a single prediction-label pair.

        Args:
            pred (torch.Tensor): Predicted logits or probabilities (C, H, W). Should be on self.device or moved.
            label (torch.Tensor): Ground truth label map (H, W), LongTensor. Should be on self.device or moved.
        """

        # Get hard predictions
        pred_hard = torch.argmax(pred.squeeze(0), dim=0) # (H, W)

        # One-hot encode
        label_onehot = F.one_hot(label.squeeze(0), num_classes=self.num_classes).permute(2, 0, 1).bool() # (C, H, W)
        pred_onehot = F.one_hot(pred_hard, num_classes=self.num_classes).permute(2, 0, 1).bool() # (C, H, W)

        # Calculate TP, FP, FN, TN per class
        tp = (pred_onehot & label_onehot).sum(dim=(1, 2))
        fp = (pred_onehot & ~label_onehot).sum(dim=(1, 2))
        fn = (~pred_onehot & label_onehot).sum(dim=(1, 2))
        tn = (~pred_onehot & ~label_onehot).sum(dim=(1, 2))
        
        # tp = (pred_onehot & label_onehot).sum(dim=(1, 2))
        # fp = pred_onehot.sum(dim=(1, 2)) - tp
        # fn = label_onehot.sum(dim=(1, 2)) - tp
        # tn = label.numel() - fn - fp - tp

        # Accumulate on CPU with float64
        self.total_tp += tp.cpu().to(torch.float64)
        self.total_fp += fp.cpu().to(torch.float64)
        self.total_fn += fn.cpu().to(torch.float64)
        self.total_tn += tn.cpu().to(torch.float64) # Accumulate TN if needed for accuracy


    def compute_epoch_metrics(self, epsilon: float = 1e-6):
        """
        Computes the macro-averaged metrics for the accumulated epoch statistics,
        appends them to the history lists, and returns the computed mean metrics.

        Args:
            epsilon (float): Small value to avoid division by zero.

        Returns:
            tuple: (mean_dice, mean_iou, mean_acc) for the current epoch.
        """

        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn
        tn = self.total_tn

        per_class_iou = tp / (tp + fp + fn)
        per_class_dice = (2 * tp) / (2 * tp + fp + fn)
        per_class_acc = (tp + tn) / (tp + tn + fp + fn)

        # Compute the mean IoU, Dice, and accuracy, considering the mask.
        mean_iou = per_class_iou[self.mask].mean().item()
        mean_dice = per_class_dice[self.mask].mean().item()
        mean_acc = per_class_acc[self.mask].mean().item()

        # Append to history
        self.epoch_mean_iou_history.append(mean_iou)
        self.epoch_mean_dice_history.append(mean_dice)
        self.epoch_mean_acc_history.append(mean_acc)

        self.epoch_per_class_iou_history.append(per_class_iou.numpy())
        self.epoch_per_class_dice_history.append(per_class_dice.numpy())
        self.epoch_per_class_acc_history.append(per_class_acc.numpy())

        self.last_per_class_iou = per_class_iou
        self.last_per_class_dice = per_class_dice
        self.last_per_class_acc = per_class_acc

        return mean_dice, mean_iou, mean_acc
      
    def to(self, device):
        """
        Moves the internal tensors to the specified device.
        Args:
            device (str): The device to move the tensors to (e.g., 'cuda', 'cpu').
        """
        self.total_tp = self.total_tp.to(device)
        self.total_fp = self.total_fp.to(device)
        self.total_fn = self.total_fn.to(device)
        self.total_tn = self.total_tn.to(device)

        self.mask = self.mask.to(device)

        if self.last_per_class_iou is not None:
            self.last_per_class_iou = self.last_per_class_iou.to(device)

        if self.last_per_class_dice is not None:
            self.last_per_class_dice = self.last_per_class_dice.to(device)

        if self.last_per_class_acc is not None:
            self.last_per_class_acc = self.last_per_class_acc.to(device)

    def get_ignore_index(self):
        return self.ignore_index
    
    def get_num_classes(self):
        return self.num_classes

    def get_mean_dice_history(self):
        return self.epoch_mean_dice_history

    def get_mean_iou_history(self):
        return self.epoch_mean_iou_history

    def get_mean_acc_history(self):
        return self.epoch_mean_acc_history
    
    def get_class_dice_history(self):
        return self.epoch_per_class_dice_history

    def get_class_iou_history(self):
        return self.epoch_per_class_iou_history

    def get_class_acc_history(self):
        return self.epoch_per_class_acc_history

    def get_last_per_class_dice(self):
        return self.last_per_class_dice
    
    def get_last_per_class_iou(self):
        return self.last_per_class_iou
    
    def get_last_per_class_acc(self):
        return self.last_per_class_acc