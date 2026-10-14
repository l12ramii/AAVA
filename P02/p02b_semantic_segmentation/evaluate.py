"""Compara checkpoints en el mismo test; exporta tabla, métricas por imagen y figura."""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from src.dataset import PetDataset, read_manifest
from src.engine import load_checkpoint, run_epoch, write_json
from src.metrics import SegmentationMetrics
from src.visualization import comparison_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcn", required=True)
    parser.add_argument("--skip", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--manifest", default="splits/default.json")
    parser.add_argument("--out", default="results/comparison")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--examples", nargs=4, type=int, help="Cuatro índices: dos éxitos y dos fallos elegidos tras el análisis")
    args = parser.parse_args()
    torch.set_num_threads(4)
    loaded = [load_checkpoint(path, args.device) for path in (args.fcn, args.skip)]
    models, records = zip(*loaded)
    manifest = read_manifest(args.manifest)
    for name, record in zip(("fcn", "skip"), records):
        if record["config"]["model"] != name or record["manifest"] != manifest:
            parser.error("Modelo o manifiesto no coincide con el experimento.")
    keys = ("size", "seed", "batch_size", "lr", "pretrained", "stopping_settings", "training_code_sha256")
    if any(records[0]["config"][k] != records[1]["config"][k] for k in keys):
        parser.error("Comparación obligatoria: use el mismo protocolo para ambos modelos.")
    dataset = PetDataset(args.data, args.manifest, "test", records[0]["config"]["size"])
    indices = args.examples or list(range(min(4, len(dataset))))
    if any(i < 0 or i >= len(dataset) for i in indices) or len(set(indices)) != len(indices):
        parser.error("Índices de ejemplos fuera de rango o repetidos.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, per_image = [], []
    for name, model, record in zip(("fcn", "skip"), models, records):
        metrics = run_epoch(model, DataLoader(dataset, batch_size=16), args.device)
        rows.append({"model": name, "trainable_parameters": record["config"]["trainable_parameters"],
                     "selected_epoch": record["epoch"], **metrics})
    with torch.inference_mode():
        for i, (x, target) in enumerate(dataset):
            row = {"index": i, "id": dataset.ids[i]}
            for name, model in zip(("fcn", "skip"), models):
                metrics = SegmentationMetrics()
                metrics.update(model(x[None].to(args.device)).argmax(1).cpu(), target[None])
                row[name + "_iou_pet"] = metrics.compute()["iou_pet"]
            per_image.append(row)
    for filename, table in (("comparison.csv", rows), ("per_image.csv", per_image)):
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_json(out / "metrics.json", {"split": "test", "n": len(dataset), "models": rows,
                                     "example_ids": [dataset.ids[i] for i in indices]})
    comparison_figure(dataset, models, indices, args.device, out / "examples.png")
    print(rows)


if __name__ == "__main__":
    main()
