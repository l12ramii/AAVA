import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .dataset import IGNORE_INDEX
from .metrics import SegmentationMetrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, device, optimizer=None, progress=None):
    training = optimizer is not None
    model.train(training)
    metrics, total_loss, total_pixels = SegmentationMetrics(), 0.0, 0
    started = time.perf_counter()
    batches = len(loader)
    interval = max(1, (batches + 9) // 10)
    if progress:
        print(f"{progress}: 0/{batches} lotes", flush=True)
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, start=1):
            images, targets = images.to(device), targets.to(device)
            valid_count = int((targets != IGNORE_INDEX).sum())
            if valid_count == 0:
                continue  # evita NaN de una media sobre cero píxeles
            logits = model(images)
            loss_sum = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum")
            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / valid_count).backward()
                optimizer.step()
            metrics.update(logits.argmax(1), targets)
            total_loss += loss_sum.item()
            total_pixels += valid_count
            if progress and (step == 1 or step % interval == 0 or step == batches):
                elapsed = time.perf_counter() - started
                remaining = elapsed / step * (batches - step)
                print(f"{progress}: {step}/{batches} lotes ({100 * step / batches:.0f}%) | "
                      f"loss={total_loss / total_pixels:.4f} | {elapsed:.1f}s | "
                      f"restante estimado={remaining:.1f}s", flush=True)
    if total_pixels == 0:
        raise ValueError("El conjunto no contiene píxeles evaluables.")
    return {"loss": total_loss / total_pixels, **metrics.compute()}


def save_checkpoint(path, model, config, epoch, val_metrics, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Incluye el encoder: cargar el checkpoint no necesita descargar pesos de ImageNet.
    atomic_torch_save({"schema": 1, "state_dict": model.state_dict(), "config": config,
                "epoch": epoch, "validation": val_metrics, "manifest": manifest,
                "labels": {"background": 0, "pet": 1, "ignore": 255}}, path)


def load_checkpoint(path, device="cpu"):
    from .decoder import build_model
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema") != 1 or record["labels"] != {"background": 0, "pet": 1, "ignore": 255}:
        raise ValueError("Checkpoint incompatible.")
    model = build_model(record["config"]["model"], pretrained=False)
    model.load_state_dict(record["state_dict"], strict=True)
    return model.to(device).eval(), record


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def atomic_torch_save(record, path):
    """Guardar primero en el mismo directorio y reemplazar solo tras cerrar el fichero."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)


def capture_rng(generator):
    np_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [np_state[0], np_state[1].tolist(),
            np_state[2], np_state[3], np_state[4]], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, values, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            print("Aviso: cambió el número de GPUs; no se garantiza identidad numérica.")
