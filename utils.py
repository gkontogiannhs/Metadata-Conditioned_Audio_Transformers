import os
import torch
import numpy as np
import torch.nn as nn

def set_visible_gpus(gpus: str, verbose: bool = True):
    """
    Restrict which GPUs PyTorch can see by setting CUDA_VISIBLE_DEVICES.

    Args:
        gpus (str): Comma-separated GPU indices, e.g., "0,1,2,3".
        verbose (bool): If True, print the selection info.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if verbose:
        print(f"[CUDA] Visible devices set to: {gpus}")

    # Optional sanity check after setting
    torch.cuda.device_count()  # forces CUDA to reinitialize


def compute_multilabel_metrics(all_labels, all_preds, all_probs=None, verbose=True):
    """
    Compute detailed multi-label metrics + official ICBHI metrics +
    macro-averaged metrics + binary Normal-vs-Abnormal ICBHI score.

    Works for multilabel setup with two binary labels [crackle, wheeze],
    forming 4 composite classes (Normal, Crackle, Wheeze, Both).
    """

    metrics = {}

    # === Composite 4-class masks (true/pred) ===
    is_n = (all_labels[:,0]==0) & (all_labels[:,1]==0)
    is_c = (all_labels[:,0]==1) & (all_labels[:,1]==0)
    is_w = (all_labels[:,0]==0) & (all_labels[:,1]==1)
    is_b = (all_labels[:,0]==1) & (all_labels[:,1]==1)

    pr_n = (all_preds[:,0]==0) & (all_preds[:,1]==0)
    pr_c = (all_preds[:,0]==1) & (all_preds[:,1]==0)
    pr_w = (all_preds[:,0]==0) & (all_preds[:,1]==1)
    pr_b = (all_preds[:,0]==1) & (all_preds[:,1]==1)

    # === Class-wise totals ===
    Nn, Nc, Nw, Nb = is_n.sum(), is_c.sum(), is_w.sum(), is_b.sum()
    Pn = np.sum(is_n & pr_n)
    Pc = np.sum(is_c & pr_c)
    Pw = np.sum(is_w & pr_w)
    Pb = np.sum(is_b & pr_b)

    # === Official 4-class ICBHI metrics ===
    sp = Pn / (Nn + 1e-12)
    se = (Pc + Pw + Pb) / (Nc + Nw + Nb + 1e-12)
    hs = 0.5 * (sp + se)
    metrics.update({'specificity': sp, 'sensitivity': se, 'icbhi_score': hs})

    if verbose:
        print(f"[ICBHI 4-class] SPE={sp*100:.2f}% | SEN={se*100:.2f}% | HS={hs*100:.2f}%")

    # === Per-class metrics ===
    pattern_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    is_true = [is_n, is_c, is_w, is_b]
    is_pred = [pr_n, pr_c, pr_w, pr_b]
    Ps = [Pn, Pc, Pw, Pb]
    Ns = [Nn, Nc, Nw, Nb]

    total_samples = len(all_labels)
    per_class = []

    for name, tmask, pmask, P, N in zip(pattern_names, is_true, is_pred, Ps, Ns):
        TP = P
        FN = N - P
        FP = np.sum(~tmask & pmask)
        TN = total_samples - TP - FP - FN

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)         # sensitivity
        specificity = TN / (TN + FP + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        # accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)

        per_class.append({
            'name': name,
            'precision': precision,
            'sensitivity': recall,
            'specificity': specificity,
            'f1': f1,
            'support': N
        })

        metrics.update({
            f'{name}_precision': precision,
            f'{name}_sensitivity': recall,
            f'{name}_specificity': specificity,
            f'{name}_f1': f1,
        })

        # if verbose:
        #     print(f"[{name}] P={precision*100:.2f}% | R={recall*100:.2f}% | "
        #           f"Sp={specificity*100:.2f}% | F1={f1*100:.2f}%")

    # === Macro and Weighted averages ===
    precisions = [c['precision'] for c in per_class]
    sensitivities = [c['sensitivity'] for c in per_class]
    specificities = [c['specificity'] for c in per_class]
    f1s = [c['f1'] for c in per_class]
    supports = [c['support'] for c in per_class]

    metrics.update({
        'macro_precision': np.mean(precisions),
        'macro_sensitivity': np.mean(sensitivities),
        'macro_specificity': np.mean(specificities),
        'macro_f1': np.mean(f1s),
        'weighted_precision': np.average(precisions, weights=supports),
        'weighted_sensitivity': np.average(sensitivities, weights=supports),
        'weighted_specificity': np.average(specificities, weights=supports),
        'weighted_f1': np.average(f1s, weights=supports),
    })

    if verbose:
        print("\n[Macro averages]")
        print(f"P={metrics['macro_precision']*100:.4f}% | R={metrics['macro_sensitivity']*100:.4f}% | "
              f"Sp={metrics['macro_specificity']*100:.4f}% | F1={metrics['macro_f1']*100:.4f}%")

    # === Binary Normal vs Abnormal (ICBHI style) ===
    is_abn_true = ~is_n
    is_abn_pred = ~pr_n

    TP = np.sum(is_abn_true & is_abn_pred)  # correctly abnormal
    TN = np.sum(is_n & pr_n)                # correctly normal
    FP = np.sum(~is_abn_true & is_abn_pred) # predicted abnormal but actually normal
    FN = np.sum(is_abn_true & ~is_abn_pred) # predicted normal but actually abnormal

    binary_spe = TN / (TN + FP + 1e-12)
    binary_sen = TP / (TP + FN + 1e-12)
    binary_hs = 0.5 * (binary_spe + binary_sen)

    metrics.update({
        'binary_specificity': binary_spe,
        'binary_sensitivity': binary_sen,
        'binary_icbhi_score': binary_hs
    })

    if verbose:
        print("\n[Binary Normal-vs-Abnormal ICBHI]")
        print(f"SPE={binary_spe*100:.4f}% | SEN={binary_sen*100:.4f}% | HS={binary_hs*100:.4f}%")

    return metrics

def _icbhi_from_bits(all_labels, all_preds):
    """Compute official ICBHI specificity, sensitivity, score from exact class matches."""
    is_n = (all_labels[:,0]==0) & (all_labels[:,1]==0)
    is_c = (all_labels[:,0]==1) & (all_labels[:,1]==0)
    is_w = (all_labels[:,0]==0) & (all_labels[:,1]==1)
    is_b = (all_labels[:,0]==1) & (all_labels[:,1]==1)

    pr_n = (all_preds[:,0]==0) & (all_preds[:,1]==0)
    pr_c = (all_preds[:,0]==1) & (all_preds[:,1]==0)
    pr_w = (all_preds[:,0]==0) & (all_preds[:,1]==1)
    pr_b = (all_preds[:,0]==1) & (all_preds[:,1]==1)

    Nn, Nc, Nw, Nb = is_n.sum(), is_c.sum(), is_w.sum(), is_b.sum()
    Pn = np.sum(is_n & pr_n)
    Pc = np.sum(is_c & pr_c)
    Pw = np.sum(is_w & pr_w)
    Pb = np.sum(is_b & pr_b)

    sp = Pn / (Nn + 1e-12)
    se = (Pc + Pw + Pb) / (Nc + Nw + Nb + 1e-12)
    hs = 0.5 * (sp + se)
    return sp, se, hs


# def find_best_thresholds_icbhi(val_labels_ml, val_probs_ml, grid=np.linspace(0.05, 0.95, 19)):
#     """
#     Sweep thresholds on validation set to maximize official ICBHI score.
#     Returns dict with best_tC, best_tW, sp, se, hs.
#     """
#     best = {"tC": 0.5, "tW": 0.5, "specificity": 0.0, "sensitivity": 0.0, "icbhi_score": 0.0}
#     y_true = val_labels_ml.astype(int)

#     for tC in grid:
#         for tW in grid:
#             y_pred = np.stack([
#                 (val_probs_ml[:,0] >= tC).astype(int),
#                 (val_probs_ml[:,1] >= tW).astype(int)
#             ], axis=1)
#             sp, se, hs = _icbhi_from_bits(y_true, y_pred)
#             if hs > best["icbhi_score"]:
#                 best.update({"tC": float(tC), "tW": float(tW),
#                              "specificity": float(sp), "sensitivity": float(se), "icbhi_score": float(hs)})
#     return best

def find_best_thresholds_icbhi(val_labels_ml, val_probs_ml, 
                                grid=np.linspace(0.1, 0.6, 11),
                                min_sensitivity=0.45):
    """Threshold search with sensitivity constraint."""
    best = {"tC": 0.3, "tW": 0.3, "sp": 0.0, "se": 0.0, "hs": 0.0}
    
    for tC in grid:
        for tW in grid:
            y_pred = np.stack([
                (val_probs_ml[:, 0] >= tC).astype(int),
                (val_probs_ml[:, 1] >= tW).astype(int)
            ], axis=1)
            sp, se, hs = _icbhi_from_bits(val_labels_ml.astype(int), y_pred)
            
            # Only accept if sensitivity meets minimum
            if se >= min_sensitivity and hs > best["hs"]:
                best.update({"tC": float(tC), "tW": float(tW),
                            "sp": float(sp), "se": float(se), "hs": float(hs)})
    
    # Fallback if no threshold meets constraint
    if best["hs"] == 0.0:
        print(f"[WARN] No threshold met min_sensitivity={min_sensitivity}, using lowest thresholds")
        best["tC"], best["tW"] = grid[0], grid[0]
    
    return best


class AsymmetricBCEWithLogitsLoss(nn.Module):
    """
    Asymmetric loss that penalizes false negatives more than false positives.
    
    Args:
        gamma_neg: focusing parameter for negatives (higher = less focus on easy negatives)
        gamma_pos: focusing parameter for positives (lower = more focus on all positives)
        pos_weight: weight for positive samples (computed from data)
        clip: probability clipping to prevent log(0)
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, pos_weight=None, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.pos_weight = pos_weight
        self.clip = clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Clip probabilities for numerical stability
        probs_pos = probs
        probs_neg = (1 - probs).clamp(min=self.clip)
        
        # Asymmetric focusing
        # For positives: mild focusing (gamma_pos=1 means standard BCE)
        # For negatives: strong focusing (gamma_neg=4 means ignore easy negatives)
        pos_term = targets * torch.log(probs_pos.clamp(min=1e-8)) * ((1 - probs_pos) ** self.gamma_pos)
        neg_term = (1 - targets) * torch.log(probs_neg) * (probs ** self.gamma_neg)
        
        loss = -pos_term - neg_term
        
        # Apply pos_weight if provided
        if self.pos_weight is not None:
            weight = targets * self.pos_weight + (1 - targets)
            loss = loss * weight
        
        return loss.mean()


class HierarchicalMultiLabelLoss(nn.Module):
    """
    Alternative: BCE-based loss with class balancing and semantic adjustments.
    
    Args:
        class_counts: [N_normal, N_crackle, N_wheeze, N_both]
        sensitivity_bias: multiplier for abnormal classes
        partial_credit: discount for Both → partial prediction
        miss_penalty: extra penalty for Abnormal → Normal
    """
    def __init__(
        self,
        class_counts=[2063, 1215, 501, 363],
        sensitivity_bias=1.5,
        partial_credit=0.5,
        miss_penalty=2.0,
    ):
        super().__init__()
        
        self.partial_credit = partial_credit
        self.miss_penalty = miss_penalty
        
        # Compute per-label pos_weight from class counts
        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            # N, C, W, B = counts
            n_total = counts.sum()
            
            # Crackle positive = C + B, negative = N + W
            n_crackle_pos = counts[1] + counts[3]
            n_crackle_neg = counts[0] + counts[2]
            
            # Wheeze positive = W + B, negative = N + C
            n_wheeze_pos = counts[2] + counts[3]
            n_wheeze_neg = counts[0] + counts[1]
            
            pos_weight = torch.tensor([
                n_crackle_neg / (n_crackle_pos + 1e-6),
                n_wheeze_neg / (n_wheeze_pos + 1e-6),
            ]) * sensitivity_bias
            
            print(f"\n[HierarchicalMultiLabelLoss] pos_weight:")
            print(f"  Crackle: {pos_weight[0]:.3f} (pos={int(n_crackle_pos)}, neg={int(n_crackle_neg)})")
            print(f"  Wheeze:  {pos_weight[1]:.3f} (pos={int(n_wheeze_pos)}, neg={int(n_wheeze_neg)})")
        else:
            pos_weight = torch.ones(2)
            
        self.register_buffer('pos_weight', pos_weight)
        
    def forward(self, logits, targets):
        # Ensure buffers on correct device
        pos_weight = self.pos_weight.to(logits.device)
        
        probs = torch.sigmoid(logits)
        
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=pos_weight,  # Use local variable
            reduction='none'
        )
        
        # Semantic adjustments
        is_both_true = (targets[:, 0] == 1) & (targets[:, 1] == 1)
        is_abnormal_true = (targets[:, 0] == 1) | (targets[:, 1] == 1)
        
        pred_c = probs[:, 0] > 0.5
        pred_w = probs[:, 1] > 0.5
        is_normal_pred = (~pred_c) & (~pred_w)
        
        # Weight adjustments
        weights = torch.ones_like(bce)
        
        # Partial credit: Both → Crackle/Wheeze
        partial_mask = is_both_true & (pred_c ^ pred_w)
        weights[partial_mask] *= self.partial_credit
        
        # Miss penalty: Abnormal → Normal  
        miss_mask = is_abnormal_true & is_normal_pred
        weights[miss_mask.unsqueeze(1).expand(-1, 2)] *= self.miss_penalty
        
        loss = (bce * weights).mean()
        
        return loss
    

class CompositeClassLoss(nn.Module):
    def __init__(
        self, 
        class_counts=None,
        sensitivity_bias=1.5,
        partial_credit=0.5,
        over_pred_cost=0.3,
        normalize_weights=True,
    ):
        super().__init__()
        
        self.sensitivity_bias = sensitivity_bias
        
        base_cost = torch.tensor([
            [0.0,  1.0,  1.0,  1.5],
            [2.0,  0.0,  1.2,  over_pred_cost],
            [2.0,  1.2,  0.0,  over_pred_cost],
            [2.5,  partial_credit, partial_credit,  0.0],
        ], dtype=torch.float32)
        
        if class_counts is not None:
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
            n_total = class_counts.sum()
            n_classes = len(class_counts)
            
            class_weights = n_total / (n_classes * class_counts + 1e-6)
            
            if normalize_weights:
                class_weights = class_weights / class_weights.sum() * n_classes
            
            class_weights[1:] *= sensitivity_bias
            
            print(f"\n[CompositeClassLoss] Class weights:")
            print(f"  Normal:  {class_weights[0]:.3f}")
            print(f"  Crackle: {class_weights[1]:.3f}")
            print(f"  Wheeze:  {class_weights[2]:.3f}")
            print(f"  Both:    {class_weights[3]:.3f}")
        else:
            class_weights = torch.ones(4)
        
        weighted_cost = base_cost * class_weights.unsqueeze(1)
        
        self.register_buffer('cost_matrix', weighted_cost)
        self.register_buffer('class_weights', class_weights)
        
    def forward(self, logits, targets):
        # Ensure cost_matrix is on same device as logits
        cost_matrix = self.cost_matrix.to(logits.device)
        
        probs = torch.sigmoid(logits)
        
        p_c, p_w = probs[:, 0], probs[:, 1]
        
        pred_probs = torch.stack([
            (1 - p_c) * (1 - p_w),
            p_c * (1 - p_w),
            (1 - p_c) * p_w,
            p_c * p_w,
        ], dim=1)
        
        # Ensure targets are on correct device and dtype
        true_class = (targets[:, 0].long() * 1 + targets[:, 1].long() * 2)
        
        costs = cost_matrix[true_class]
        
        loss = (pred_probs * costs).sum(dim=1).mean()
        
        return loss


def compute_pos_weight(dataloader, device, sensitivity_bias=2.0):
    """
    Compute pos_weight with sensitivity bias.
    
    Args:
        sensitivity_bias: multiplier to increase weight on positives (default 2.0)
                         Higher = more penalty for missing positives
    """
    all_labels = []
    
    for batch in dataloader:
        labels = batch["label"].numpy()
        all_labels.append(labels)
    
    all_labels = np.vstack(all_labels)
    n_total = len(all_labels)
    
    n_pos = all_labels.sum(axis=0)
    n_neg = n_total - n_pos
    
    # Base pos_weight
    pos_weight = n_neg / (n_pos + 1e-6)
    
    # Apply sensitivity bias
    pos_weight = pos_weight * sensitivity_bias
    
    print(f"\n[Pos Weight] Label distribution:")
    print(f"  Crackle: {int(n_pos[0])}/{n_total} positive ({n_pos[0]/n_total*100:.1f}%)")
    print(f"  Wheeze:  {int(n_pos[1])}/{n_total} positive ({n_pos[1]/n_total*100:.1f}%)")
    print(f"  Base pos_weight: [{n_neg[0]/n_pos[0]:.2f}, {n_neg[1]/n_pos[1]:.2f}]")
    print(f"  With bias ({sensitivity_bias}x): [{pos_weight[0]:.2f}, {pos_weight[1]:.2f}]")
    
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


class CompositeClassLoss(nn.Module):
    def __init__(
        self, 
        class_counts=[2063, 1215, 501, 363],
        sensitivity_bias=1.5,
        partial_credit=0.5,
        over_pred_cost=0.3,
        normalize_weights=True,
    ):
        super().__init__()
        
        self.sensitivity_bias = sensitivity_bias
        
        base_cost = torch.tensor([
            [0.0,  1.0,  1.0,  1.5],
            [2.0,  0.0,  1.2,  over_pred_cost],
            [2.0,  1.2,  0.0,  over_pred_cost],
            [2.5,  partial_credit, partial_credit,  0.0],
        ], dtype=torch.float32)  # Explicit dtype
        
        if class_counts is not None:
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
            n_total = class_counts.sum()
            n_classes = len(class_counts)
            
            class_weights = n_total / (n_classes * class_counts + 1e-6)
            
            if normalize_weights:
                class_weights = class_weights / class_weights.sum() * n_classes
            
            class_weights[1:] *= sensitivity_bias
            
            print(f"\n[CompositeClassLoss] Class weights:")
            print(f"  Normal:  {class_weights[0]:.3f}")
            print(f"  Crackle: {class_weights[1]:.3f}")
            print(f"  Wheeze:  {class_weights[2]:.3f}")
            print(f"  Both:    {class_weights[3]:.3f}")
        else:
            class_weights = torch.ones(4)
        
        weighted_cost = base_cost * class_weights.unsqueeze(1)
        
        self.register_buffer('cost_matrix', weighted_cost)
        self.register_buffer('class_weights', class_weights)
        
    def forward(self, logits, targets):
        # Ensure cost_matrix is on same device as logits
        cost_matrix = self.cost_matrix.to(logits.device)
        
        probs = torch.sigmoid(logits)
        
        p_c, p_w = probs[:, 0], probs[:, 1]
        
        pred_probs = torch.stack([
            (1 - p_c) * (1 - p_w),
            p_c * (1 - p_w),
            (1 - p_c) * p_w,
            p_c * p_w,
        ], dim=1)
        
        # Ensure targets are on correct device and dtype
        true_class = (targets[:, 0].long() * 1 + targets[:, 1].long() * 2)
        
        costs = cost_matrix[true_class]
        
        loss = (pred_probs * costs).sum(dim=1).mean()
        
        return loss