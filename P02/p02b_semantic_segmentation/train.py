"""Entrenamiento con criterio T7 y reanudación desde la última época completa."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from src.dataset import PetDataset, read_manifest
from src.decoder import build_model
from src.engine import (seed_everything, run_epoch, save_checkpoint, write_json,
                        atomic_torch_save, capture_rng, restore_rng)
from src.stopping import should_stop
from src.visualization import plot_history


def code_hash():
    root = Path(__file__).resolve().parent
    files = [root / "train.py", *sorted((root / "src").glob("*.py"))]
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()


def stop_decision(history, settings):
    decision = should_stop(history, settings)
    if (not isinstance(decision, tuple) or len(decision) != 2 or type(decision[0]) is not bool
            or not isinstance(decision[1], str) or (decision[0] and not decision[1].strip())
            or (not decision[0] and decision[1] != "")):
        raise ValueError("T7 debe devolver (bool, motivo): (False, '') o (True, texto no vacío).")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["fcn", "skip"], required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--manifest", default="splits/default.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20, help="Presupuesto TOTAL de épocas, ampliable; no épocas extra")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stopping", default="stopping.json")
    parser.add_argument("--resume", action="store_true", help="Recuperar <out>/last.pt; mantener código y protocolo")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.lr <= 0 or args.size < 32:
        parser.error("Épocas, batch, threads y lr positivos; tamaño >=32.")
    out = Path(args.out)
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error("Salida no vacía: usa --resume o una carpeta nueva.")
    settings = json.loads(Path(args.stopping).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or not isinstance(settings.get("description"), str) or not settings["description"].strip():
        parser.error("stopping.json requiere description no vacía y los parámetros de tu regla.")
    stop_decision([], settings)  # contrato de historial vacío; falla pronto si T7 está pendiente
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA no disponible; usa --device cpu o auto.")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    manifest = read_manifest(args.manifest)
    generator = torch.Generator().manual_seed(args.seed)
    train = PetDataset(args.data, args.manifest, "train", args.size, augment=True)
    val = PetDataset(args.data, args.manifest, "val", args.size)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    state = None
    if args.resume:
        if not (out / "last.pt").is_file():
            parser.error("No existe last.pt: antes de la primera época completa usa una carpeta nueva.")
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if state.get("resume_schema") != 1 or state["manifest"] != manifest:
            parser.error("Checkpoint de reanudación o manifiesto incompatible.")
    model = build_model(args.model, pretrained=state is None).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    config = {**vars(args), "device": device, "pretrained": "ResNet18_IMAGENET1K_V1",
              "python": platform.python_version(), "torch": str(torch.__version__),
              "torchvision": str(torchvision.__version__), "stopping_settings": settings,
              "training_code_sha256": code_hash(),
              "trainable_parameters": sum(p.numel() for p in parameters)}
    history, best_record = [], None
    if state is not None:
        keys = ("model", "size", "seed", "batch_size", "lr", "pretrained", "threads",
                "stopping_settings", "training_code_sha256", "torch", "torchvision")
        if any(state["config"][k] != config[k] for k in keys):
            parser.error("Cambió código, regla, entorno o protocolo: inicia un experimento nuevo.")
        if args.epochs < state["epoch"]:
            parser.error("--epochs es total y no puede ser menor que la época recuperada.")
        if state["config"]["device"] != device:
            print("Aviso: cambió dispositivo; se recupera estado, sin garantía de identidad numérica.")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        history, best_record = state["history"], state["best_record"]
        restore_rng(state["rng"], generator)
        # last.pt es la autoridad: repara exportaciones si hubo corte entre escrituras.
        atomic_torch_save(best_record, out / "best.pt")
        print(f"Recuperada época {state['epoch']}; se continúa desde {state['epoch'] + 1}.", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "manifest.json", manifest)
    write_json(out / "history.json", history)
    stopped, reason = stop_decision(history, settings)
    print(f"{args.model}: {device}; {len(train)}/{len(val)} imágenes; presupuesto total {args.epochs} épocas.", flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        if stopped:
            break
        start = time.perf_counter()
        tr = run_epoch(model, train_loader, device, optimizer, progress=f"Época {epoch} train")
        va = run_epoch(model, val_loader, device, progress=f"Época {epoch} val")
        history.append({"epoch": epoch, "train": tr, "val": va, "seconds": time.perf_counter() - start})
        stopped, reason = stop_decision(history, settings)
        if best_record is None or va["miou"] > best_record["validation"]["miou"]:
            # Construye el checkpoint de inferencia en CPU, sin referencias a pesos mutables.
            best_record = {"schema": 1, "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                           "config": config.copy(), "epoch": epoch, "validation": va, "manifest": manifest,
                           "labels": {"background": 0, "pet": 1, "ignore": 255}}
        resume = {"resume_schema": 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": epoch, "history": history, "best_record": best_record,
                  "config": config, "manifest": manifest, "rng": capture_rng(generator)}
        atomic_torch_save(resume, out / "last.pt")
        atomic_torch_save(best_record, out / "best.pt")
        write_json(out / "history.json", history)
        plot_history(history, out / "curves.png")
        print(f"Época {epoch}: val mIoU={va['miou']:.4f}; last.pt guardado.", flush=True)
    if not stopped:
        reason = "Presupuesto total de épocas alcanzado"
    summary = {"completed_epochs": len(history), "stop_reason": reason,
               "criterion_triggered": stopped, "budget_epochs": args.epochs,
               "selected_epoch": best_record["epoch"] if best_record else None,
               "training_seconds": sum(r["seconds"] for r in history)}
    write_json(out / "summary.json", summary)
    if history:
        plot_history(history, out / "curves.png")
    print(summary)


if __name__ == "__main__":
    main()
