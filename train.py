import time
import datetime
import torch
import torch.optim as optim
import pandas as pd
import yaml
import io

from contextlib import redirect_stdout
from tqdm import tqdm
from pathlib import Path

from torch.optim.lr_scheduler import StepLR
from dataset import build_dataloader
from model import build_model

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_epochs):
    """Train a detection model for a single epoch."""
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=True)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / n:.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def collect_detections(model, data_loader, device, label_to_coco_id,
                       score_thr=0.05, desc="Inference"):
    model.eval()
    results = []
    image_ids = set()

    for images, targets in tqdm(data_loader, desc=desc, leave=False):
        images = [img.to(device) for img in images]
        preds = model(images)

        for img_idx, pred in enumerate(preds):
            img_id = int(targets[img_idx]["image_id"].item())
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                li = int(label)
                if score < score_thr or li not in label_to_coco_id:
                    continue
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": img_id,
                    "category_id": int(label_to_coco_id[li]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
            image_ids.add(img_id)

    return results, sorted(image_ids)


def _iou_xywh(a, b):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes
    represented in `[x, y, w, h]` format.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def compute_prf1(coco_gt, results, iou_thr=0.5, score_thr=0.5):
    """Compute precision, recall, and F1-score."""
    gt_index = {}
    for ann in coco_gt.dataset["annotations"]:
        gt_index.setdefault(ann["image_id"], []).append({
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
            "matched": False,
        })

    preds = [r for r in results if r["score"] >= score_thr]
    preds.sort(key=lambda r: r["score"], reverse=True)

    tp = fp = 0
    for pred in preds:
        gts = gt_index.get(pred["image_id"], [])
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gts):
            if gt["matched"] or gt["category_id"] != pred["category_id"]:
                continue
            iou = _iou_xywh(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_thr and best_idx >= 0:
            gts[best_idx]["matched"] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for gts in gt_index.values() for gt in gts if not gt["matched"])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


@torch.no_grad()
def evaluate_coco(model, data_loader, device, coco_gt, label_to_coco_id,
                  score_thr=0.05, iou_thr=0.5, prf1_score_thr=0.5,
                  desc="Eval", verbose=False):
    """
    Evaluate an object detection model on a COCO-style dataset and compute
    COCO mAP metrics plus precision, recall, and F1-score.
    """
    results, image_ids = collect_detections(
        model, data_loader, device, label_to_coco_id,
        score_thr=score_thr, desc=desc,
    )

    empty = {
        "mAP_0.5": 0.0, "mAP_0.75": 0.0, "mAP_0.5_0.95": 0.0,
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }

    if not results:
        if verbose:
            print(f"[{desc}] no detections.")
        return empty

    # COCO mAP
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = sorted(set(int(v) for v in label_to_coco_id.values()))

    coco_eval.evaluate()
    coco_eval.accumulate()

    if verbose:
        coco_eval.summarize()
    else:
        with redirect_stdout(io.StringIO()):
            coco_eval.summarize()

    # Precision/Recall/F1
    p, r, f1 = compute_prf1(coco_gt, results,
                            iou_thr=iou_thr, score_thr=prf1_score_thr)

    return {
        "mAP_0.5": float(coco_eval.stats[1]),
        "mAP_0.75": float(coco_eval.stats[2]),
        "mAP_0.5_0.95": float(coco_eval.stats[0]),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloader(cfg)

    train_ann = resolve_path(cfg["dataset"]["train"]) / cfg["dataset"]["ann_file"]
    val_ann = resolve_path(cfg["dataset"]["valid"]) / cfg["dataset"]["ann_file"]
    coco_train = COCO(train_ann)
    coco_gt = COCO(val_ann)

    print("\n=== Dataset info ===")
    print(f"Train categories : {coco_train.dataset['categories']}")
    print(f"Val categories   : {coco_gt.dataset['categories']}")
    print(f"Val ann cat ids  : {sorted({a['category_id'] for a in coco_gt.dataset['annotations']})}")
    print("====================\n")

    tcfg = cfg["training"]
    architecture = "frcnn_mobilenet"
    n_categories = len(coco_train.dataset["categories"])
    num_classes = n_categories + 1
    print(f"Architecture : {architecture}")
    print(f"Num classes  : {num_classes}")

    model = build_model(num_classes, architecture=architecture).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=tcfg["optimizer"]["lr"])
    scheduler = StepLR(optimizer,
                       step_size=tcfg["scheduler"]["step_size"],
                       gamma=tcfg["scheduler"]["gamma"])
    num_epochs = tcfg["epochs"]

    train_cats = coco_train.dataset["categories"]
    val_cats = coco_gt.dataset["categories"]
    val_cats_by_name = {(c.get("name") or "").strip(): c["id"] for c in val_cats}

    label_to_coco_id = {}
    for i, tc in enumerate(train_cats):
        model_label = i + 1
        name = (tc.get("name") or "").strip()
        if name in val_cats_by_name:
            label_to_coco_id[model_label] = val_cats_by_name[name]
        elif i < len(val_cats):
            label_to_coco_id[model_label] = val_cats[i]["id"]
        else:
            raise RuntimeError(f"Cannot map model label {model_label} to a validation COCO category ID.")

    # Output 
    checkpoint_dir = resolve_path(tcfg["checkpoint_dir"])
    run_ts = datetime.datetime.now().strftime("%d%m%y_%H%M")
    output_dir = checkpoint_dir / f"run_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log_file = output_dir / "training_log.csv"
    columns = [
        "epoch", "train_loss", "val_loss",
        "precision", "recall", "f1",
        "val_precision", "val_recall", "val_f1",
        "mAP_0.5", "mAP_0.75", "mAP_0.5_0.95",
        "lr", "time_epoch_sec",
    ]
    pd.DataFrame(columns=columns).to_csv(log_file, index=False)

    best_map, best_epoch = 0.0, -1

    for epoch in range(1, num_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()


        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, num_epochs
        )


        train_metrics = evaluate_coco(
            model, train_loader, device, coco_train, label_to_coco_id,
            desc="Eval(train)", verbose=False,
        )

        val_metrics = evaluate_coco(
            model, val_loader, device, coco_gt, label_to_coco_id,
            desc="Eval(val)", verbose=True,
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        val_loss = float("nan")

        print(
            f"\nEpoch {epoch}/{num_epochs} | train_loss={train_loss:.4f} "
            f"| mAP@0.5={val_metrics['mAP_0.5']:.4f} "
            f"| mAP@0.75={val_metrics['mAP_0.75']:.4f} "
            f"| mAP@0.5:0.95={val_metrics['mAP_0.5_0.95']:.4f} "
            f"| time={dt:.1f}s | lr={lr:.6f}"
        )
        print(
            f"Train P/R/F1: {train_metrics['precision']:.4f}/"
            f"{train_metrics['recall']:.4f}/{train_metrics['f1']:.4f}  |  "
            f"Val P/R/F1: {val_metrics['precision']:.4f}/"
            f"{val_metrics['recall']:.4f}/{val_metrics['f1']:.4f}"
        )

        row = pd.DataFrame([{
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "precision":     train_metrics["precision"],
            "recall":        train_metrics["recall"],
            "f1":            train_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall":    val_metrics["recall"],
            "val_f1":        val_metrics["f1"],
            "mAP_0.5":       val_metrics["mAP_0.5"],
            "mAP_0.75":      val_metrics["mAP_0.75"],
            "mAP_0.5_0.95":  val_metrics["mAP_0.5_0.95"],
            "lr":            lr,
            "time_epoch_sec": dt,
        }])
        row.to_csv(log_file, mode="a", header=False, index=False)

        # Save best by val mAP@0.5
        if val_metrics["mAP_0.5"] > best_map:
            best_map = val_metrics["mAP_0.5"]
            best_epoch = epoch
            torch.save({
                "num_classes": num_classes,
                "architecture": architecture,
                "model_state_dict": model.state_dict(),
                "best_map": best_map,
                "best_epoch": best_epoch,
            }, output_dir / "best.pt")
            print(f"best model saved (mAP@0.5={best_map:.4f})")

        torch.save({
            "epoch": epoch,
            "num_classes": num_classes,
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
            "best_epoch": best_epoch,
        }, output_dir / "last.pt")

    print(f"\nDone. Best mAP@0.5={best_map:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()