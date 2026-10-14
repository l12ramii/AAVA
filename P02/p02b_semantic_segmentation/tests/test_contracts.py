"""Contratos visibles T0/T5/T7 e infraestructura de selección/reanudación; sin red."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from src.dataset import (trimap_tensor, image_tensor, load_pair, PetDataset, OriginalPetDataset,
                         make_inference_subset, MEAN, STD)
from src.inference import predict_mask, predict_mask_labels
from src.stopping import should_stop
from src.engine import seed_everything, capture_rng, restore_rng, atomic_torch_save


def fixture_data(tmp_path, nval=30):
    base = tmp_path / "oxford-iiit-pet"
    (base / "images").mkdir(parents=True)
    (base / "annotations/trimaps").mkdir(parents=True)
    manifest = {"train": ["train0", "train1"], "val": [f"val{i}" for i in range(nval)], "test": ["test0"]}
    for name in sum(manifest.values(), []):
        pixels = np.zeros((37, 51, 3), np.uint8)
        pixels[:, :25] = [255, 0, 0]
        pixels[:, 25:] = [0, 0, 255]
        Image.fromarray(pixels).save(base / "images" / f"{name}.jpg")
        raw = np.full((37, 51), 2, np.uint8)
        raw[:, :25] = 1
        raw[:2] = 3
        Image.fromarray(raw).save(base / "annotations/trimaps" / f"{name}.png")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, base


def test_normalization_rgb_and_dtype():
    result = image_tensor(Image.new("L", (13, 9), 255), 32)
    assert result.shape == (3, 32, 32) and result.dtype == torch.float32
    for c in range(3):
        assert torch.allclose(result[c], torch.full((32, 32), (1 - MEAN[c]) / STD[c]), atol=1e-6)
    with pytest.raises(ValueError):
        trimap_tensor(Image.fromarray(np.array([[0]], np.uint8)))


def test_data_load_and_augment_contract(tmp_path, monkeypatch):
    manifest, base = fixture_data(tmp_path)
    image, trimap = load_pair(base, "train0")
    assert image.mode == "RGB" and image.size == trimap.size == (51, 37)
    assert np.array(trimap)[0, 0] == 3
    plain = PetDataset(tmp_path, manifest, "train", 32)
    augmented = PetDataset(tmp_path, manifest, "train", 32, augment=True)
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor(.1))
    x, y = plain[0]
    xf, yf = augmented[0]
    assert torch.equal(y.flip(-1), yf) and torch.equal(x.flip(-1), xf)
    val = PetDataset(tmp_path, manifest, "val", 32)
    xv, yv = val[0]
    assert torch.equal(xv, x) and torch.equal(yv, y)
    with pytest.raises(ValueError):
        PetDataset(tmp_path, manifest, "val", 32, augment=True)


def test_original_data_and_fixed_subset(tmp_path):
    manifest, _ = fixture_data(tmp_path, 40)
    subset = tmp_path / "subset.json"
    first = make_inference_subset(manifest, subset)
    assert first == make_inference_subset(manifest, subset)
    assert len(first["ids"]) == len(set(first["ids"])) == 30
    dataset = OriginalPetDataset(tmp_path, manifest, subset)
    image, target = dataset[0]
    assert image.size == (51, 37) and target.shape == (37, 51) and target.dtype == torch.long
    assert set(target.unique().tolist()) == {0, 1, 255}
    first["ids"][0] = "test0"
    subset.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError):
        OriginalPetDataset(tmp_path, manifest, subset)


def test_inference_methods_order():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[[[0., 0.]], [[-1., 11.]]]], device=x.device).expand(x.shape[0], -1, -1, -1)
    model = Fixed().train()
    image = Image.new("RGB", (5, 3))
    a = predict_mask(model, image, 32)
    b = predict_mask_labels(model, image, 32)
    assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape == (3, 5)
    assert a[0].tolist() == [0, 1, 1, 1, 1]
    assert b[0].tolist() == [0, 0, 1, 1, 1]
    assert not model.training


def test_stopping_contract_purity():
    settings = json.loads(Path("stopping.json").read_text(encoding="utf-8"))
    histories = [[], [{"epoch": i, "train": {"loss": 1 / i, "miou": .4, "iou_background": .4, "iou_pet": .4, "pixel_accuracy": .6},
                      "val": {"loss": 1., "miou": .4, "iou_background": .4, "iou_pet": .4, "pixel_accuracy": .6}, "seconds": 1.} for i in range(1, 9)]]
    for history in histories:
        original_history, original_settings = copy.deepcopy(history), copy.deepcopy(settings)
        result = should_stop(history, settings)
        assert isinstance(result, tuple) and len(result) == 2 and type(result[0]) is bool
        assert isinstance(result[1], str)
        assert bool(result[1].strip()) if result[0] else result[1] == ""
        assert result == should_stop(history, settings)
        assert history == original_history and settings == original_settings
    assert should_stop([], settings) == (False, "")


def test_resume_rng_and_atomic_file(tmp_path):
    seed_everything(2026)
    generator = torch.Generator().manual_seed(2026)
    path = tmp_path / "last.pt"
    atomic_torch_save({"rng": capture_rng(generator)}, path)
    expected = (random.random(), np.random.rand(), torch.rand(3), torch.randperm(10, generator=generator))
    state = torch.load(path, weights_only=True)
    restore_rng(state["rng"], generator)
    actual = (random.random(), np.random.rand(), torch.rand(3), torch.randperm(10, generator=generator))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2]) and torch.equal(actual[3], expected[3])
    assert not path.with_name("last.pt.tmp").exists()
