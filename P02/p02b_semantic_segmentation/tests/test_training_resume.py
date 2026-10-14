"""Verifica igualdad de entrenamiento continuo/reanudado, sin descargar pesos."""
import json
import sys
import torch
import pytest
from test_contracts import fixture_data
import train
from src.decoder import build_model as actual_build_model


@pytest.mark.parametrize("name", ["fcn", "skip"])
def test_training_resume_matches_continuous(tmp_path, monkeypatch, name):
    manifest, _ = fixture_data(tmp_path / "dataset", nval=2)
    settings = tmp_path / "rule.json"
    settings.write_text(json.dumps({"description": "Referencia para prueba corta", "min_epochs": 5,
                                    "patience": 3, "min_delta": .002}), encoding="utf-8")
    # Probar el flujo real con encoder no preentrenado: no necesita Internet.
    monkeypatch.setattr(train, "build_model", lambda model, pretrained=True: actual_build_model(model, False))
    monkeypatch.setattr(train, "should_stop", lambda history, settings: (False, ""))
    full, split = tmp_path / "full", tmp_path / "split"
    def run(out, epochs, resume=False):
        argv = ["train.py", "--model", name, "--data", str(manifest.parent), "--manifest", str(manifest),
                "--out", str(out), "--size", "32", "--batch-size", "2", "--threads", "2",
                "--epochs", str(epochs), "--device", "cpu", "--stopping", str(settings)]
        if resume:
            argv.append("--resume")
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
    run(full, 2)
    run(split, 1)
    run(split, 2, True)
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(split / "last.pt", weights_only=True)
    assert a["epoch"] == b["epoch"] == 2
    assert all(torch.equal(v, b["state_dict"][k]) for k, v in a["state_dict"].items())
    for ra, rb in zip(a["history"], b["history"]):
        assert ra["train"] == rb["train"] and ra["val"] == rb["val"]
    before = {k: v.clone() for k, v in b["state_dict"].items()}
    run(split, 2, True)  # ejecutar otra vez no añade épocas ni actualiza pesos
    c = torch.load(split / "last.pt", weights_only=True)
    assert all(torch.equal(v, c["state_dict"][k]) for k, v in before.items())
    # Cambiar la política no puede continuar el mismo experimento.
    content = json.loads(settings.read_text())
    content["description"] = "Regla cambiada"
    settings.write_text(json.dumps(content))
    with pytest.raises(SystemExit):
        run(split, 3, True)
