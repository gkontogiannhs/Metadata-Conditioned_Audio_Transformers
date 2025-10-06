import numpy as np
import torch

from ls.metrics import compute_classification_metrics


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a given loader and return loss + metrics.
    """
    model.eval()
    total_loss, n_samples = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    iters = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch["input_values"].to(device), batch["labels"].to(device)
            logits = model(x)
            loss = criterion(logits, y) if criterion else 0.0

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            
            if iters == 2:
                break
            iters += 1
    avg_loss = total_loss / n_samples
    n_classes = logits.shape[1]

    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), n_classes=n_classes
    )

    return avg_loss, metrics

    # # Pretty print
    # print(f"[{prefix}][Epoch {epoch}] "
    #       f"Loss={avg_loss:.4f} | "
    #       f"Acc={metrics['accuracy']:.2f} | "
    #       f"BalAcc={metrics['balanced_acc']:.2f} | "
    #       f"F1 (Macro)={metrics['f1_macro']:.2f} | "
    #       f"S_p={metrics['specificity']:.2f} | S_e={metrics['sensitivity']:.2f} | "
    #       f"ICBHI Score={metrics['icbhi_score']:.2f}")

    # return avg_loss, metrics