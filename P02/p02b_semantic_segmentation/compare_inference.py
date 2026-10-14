"""Experimento obligatorio: dos métodos x dos modelos, GT original, validación fija."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.dataset import OriginalPetDataset, read_manifest
from src.engine import load_checkpoint, write_json
from src.inference import predict_mask, predict_mask_labels
from src.metrics import SegmentationMetrics
from src.visualization import mask_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--manifest", default="splits/default.json")
    parser.add_argument("--subset", default="splits/inference.json")
    parser.add_argument("--fcn", required=True)
    parser.add_argument("--skip", required=True)
    parser.add_argument("--out", default="results/inference")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = OriginalPetDataset(args.data, args.manifest, args.subset)
    loaded = [load_checkpoint(path, args.device) for path in (args.fcn, args.skip)]
    manifest = read_manifest(args.manifest)
    keys = ("size", "seed", "batch_size", "lr", "pretrained", "stopping_settings", "training_code_sha256")
    if any(record["manifest"] != manifest or record["config"]["model"] != name
           for name, (_, record) in zip(("fcn", "skip"), loaded)):
        parser.error("Checkpoints de modelo/manifiesto incorrectos.")
    if any(loaded[0][1]["config"][k] != loaded[1][1]["config"][k] for k in keys):
        parser.error("Los modelos no comparten protocolo y regla de parada.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, per_image, panels = [], [], []
    for name, (model, record) in zip(("fcn", "skip"), loaded):
        meters = {method: SegmentationMetrics() for method in ("logits", "labels")}
        seconds = {method: 0.0 for method in meters}
        changed, valid_total = 0, 0
        # Calentamiento fuera de tiempos; no cambia métricas ni pesos.
        warm_image, _ = dataset[0]
        for fn in (predict_mask, predict_mask_labels):
            fn(model, warm_image, record["config"]["size"], args.device)
        for index in range(len(dataset)):
            image, target = dataset[index]
            predictions, image_scores = {}, {}
            for method, fn in (("logits", predict_mask), ("labels", predict_mask_labels)):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = fn(model, image, record["config"]["size"], args.device)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                seconds[method] += elapsed
                prediction = torch.from_numpy(prediction.astype(np.int64))
                meters[method].update(prediction, target)
                local = SegmentationMetrics()
                local.update(prediction, target)
                image_scores[method] = local.compute()["iou_pet"]
                predictions[method] = prediction
            valid = target != 255
            n_valid = int(valid.sum())
            n_changed = int(((predictions["logits"] != predictions["labels"]) & valid).sum())
            changed += n_changed
            valid_total += n_valid
            per_image.append({"model": name, "id": dataset.ids[index],
                              "logits_iou_pet": image_scores["logits"], "labels_iou_pet": image_scores["labels"],
                              "changed_valid_fraction": n_changed / n_valid if n_valid else None})
            if index < 2:
                panels.append((name, dataset.ids[index], image, target, predictions))
        if not valid_total:
            raise ValueError("No hay píxeles válidos a resolución original.")
        for method in meters:
            rows.append({"model": name, "method": method, "n_images": len(dataset),
                         **meters[method].compute(), "seconds_per_image": seconds[method] / len(dataset),
                         "changed_valid_fraction": changed / valid_total})
    for filename, table in (("comparison.csv", rows), ("per_image.csv", per_image)):
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_json(out / "metrics.json", {"split": "val", "resolution": "original", "models": rows,
                                     "ids": dataset.ids, "timing": "preprocessing+modelo+resize, sin lectura ni calentamiento"})
    write_json(out / "subset.json", json.loads(Path(args.subset).read_text(encoding="utf-8")))
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 3 * len(panels)))
    axes = fig.subplots(len(panels), 4, squeeze=False)
    for row, (name, identifier, image, target, predictions) in enumerate(panels):
        for col, (title, panel) in enumerate(zip(("Original", "GT: magenta=ignorado", "Resize logits -> argmax", "Argmax -> nearest"),
                                                 (image, mask_rgb(target), mask_rgb(predictions["logits"]), mask_rgb(predictions["labels"])))):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(title + (f"\n{name}: {identifier}" if col == 0 else ""), fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=140)
    print(rows)


if __name__ == "__main__":
    main()
