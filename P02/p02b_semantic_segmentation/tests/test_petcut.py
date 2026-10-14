"""Pruebas sin red ni dataset: ejecutar python -m pytest -q."""
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import paired_transform, read_manifest
from src.decoder import build_model, DecoderBlock
from src.engine import run_epoch, save_checkpoint, load_checkpoint
from src.metrics import SegmentationMetrics
from src.inference import predict_mask, apply_effect, PetCut

torch.set_num_threads(2)


def test_trimap_mapping_and_nearest():
    mask = Image.fromarray(np.array([[1, 2], [3, 1]], dtype=np.uint8))
    image = Image.new("RGB", (2, 2))
    _, target = paired_transform(image, mask, size=4)
    assert target.dtype == torch.int64
    assert target.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [255, 255, 1, 1], [255, 255, 1, 1]]


def test_joint_flip():
    pixels = np.array([[[255, 0, 0], [0, 0, 255]], [[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    mask = Image.fromarray(np.array([[1, 2], [1, 2]], dtype=np.uint8))
    x, y = paired_transform(Image.fromarray(pixels), mask, 2)
    xf, yf = paired_transform(Image.fromarray(pixels), mask, 2, flip=True)
    assert torch.equal(x.flip(-1), xf)
    assert torch.equal(y.flip(-1), yf)


def test_metrics_ignored_pixels_and_global_aggregation():
    meter = SegmentationMetrics()
    meter.update(torch.tensor([0, 1, 1, 0, 1]), torch.tensor([0, 0, 1, 1, 255]))
    meter.update(torch.tensor([1, 1]), torch.tensor([1, 1]))
    assert meter.confusion.tolist() == [[1, 1], [1, 3]]
    result = meter.compute()
    assert result["iou_background"] == pytest.approx(1 / 3)
    assert result["iou_pet"] == pytest.approx(3 / 5)
    assert result["miou"] == pytest.approx((1 / 3 + 3 / 5) / 2)
    assert result["pixel_accuracy"] == pytest.approx(4 / 6)


def test_metrics_empty_and_absent_class():
    meter = SegmentationMetrics()
    meter.update(torch.tensor([1]), torch.tensor([255]))
    assert all(v is None for v in meter.compute().values())
    meter.update(torch.tensor([0]), torch.tensor([0]))
    assert meter.compute() == {"iou_background": 1.0, "iou_pet": None, "miou": 1.0, "pixel_accuracy": 1.0}


def test_decoder_odd_shapes_and_skip_gradient():
    block = DecoderBlock(8, 4, 3)
    deep, skip = torch.randn(2, 8, 3, 4, requires_grad=True), torch.randn(2, 4, 7, 9, requires_grad=True)
    result = block(deep, skip)
    assert result.shape == (2, 3, 7, 9)
    result.square().mean().backward()
    assert deep.grad.abs().sum() > 0 and skip.grad.abs().sum() > 0


def test_fcn_logits_shape():
    model = build_model("fcn", pretrained=False)
    logits = model(torch.randn(2, 3, 65, 79))
    assert logits.shape == (2, 2, 65, 79)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1026


@pytest.mark.parametrize("name", ["fcn", "skip"])
def test_models_update_only_decoder(name):
    model = build_model(name, pretrained=False)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    head_before = model.classifier.weight.detach().clone()
    images, targets = torch.randn(2, 3, 65, 79), torch.randint(0, 2, (2, 65, 79))
    targets[:, :2, :] = 255
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    scores = run_epoch(model, DataLoader(TensorDataset(images, targets), batch_size=2), "cpu", optimizer)
    assert np.isfinite(scores["loss"])
    assert not model.encoder.training
    assert all(torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items())
    assert all(p.grad is None for p in model.encoder.parameters())
    assert not torch.equal(head_before, model.classifier.weight)
    assert model(images).shape == (2, 2, 65, 79)


def test_all_ignored_epoch_reports_error():
    class Dummy(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("No debe ejecutar el modelo")
    data = TensorDataset(torch.zeros(1, 3, 32, 32), torch.full((1, 32, 32), 255))
    with pytest.raises(ValueError, match="evaluables"):
        run_epoch(Dummy(), DataLoader(data), "cpu")


def test_checkpoint_and_original_size(tmp_path):
    model = build_model("skip", pretrained=False).eval()
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, {"model": "skip", "size": 64}, 1, {"miou": .2}, {})
    loaded, record = load_checkpoint(path)
    image = Image.new("RGB", (91, 57), "orange")
    a, b = predict_mask(model, image, 64), predict_mask(loaded, image, 64)
    assert a.shape == (57, 91) and a.dtype == np.uint8
    assert np.array_equal(a, b) and record["epoch"] == 1
    mask, overlay, result, pair = PetCut(path)(image, "Sticker")
    assert mask.size == overlay.size == result.size == image.size
    assert result.mode == "RGBA" and len(pair) == 2


@pytest.mark.parametrize("effect", ["Desenfocar fondo", "Fondo de color", "Fondo gris", "Sticker"])
def test_effects_preserve_foreground(effect):
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4] = 1
    result = np.array(apply_effect(image, mask, effect, "#123456"))
    assert np.array_equal(result[:4, :, :3], np.array(image)[:4])
    if effect == "Sticker":
        assert np.array_equal(result[:, :, 3], mask * 255)
    elif effect == "Fondo de color":
        assert (result[4:] == [18, 52, 86]).all()
    elif effect == "Fondo gris":
        assert (result[4:, :, 0] == result[4:, :, 1]).all()


def test_manifest_rejects_overlap(tmp_path):
    path = tmp_path / "split.json"
    path.write_text('{"train":["a"],"val":["a"],"test":["b"]}')
    with pytest.raises(ValueError, match="solapadas"):
        read_manifest(path)


def test_gradio_build_and_callback(tmp_path):
    from app import build_app
    path = tmp_path / "model.pt"
    save_checkpoint(path, build_model("skip", False), {"model": "skip", "size": 32}, 1, {}, {})
    demo = build_app(path)
    callbacks = [fn.fn for fn in demo.fns.values() if len(fn.inputs) == 4 and len(fn.outputs) == 4]
    assert len(callbacks) == 1
    result = callbacks[0](Image.new("RGB", (45, 35)), "Fondo gris", "#ffffff", 12)
    assert len(result) == 4 and result[2].size == (45, 35)
    import gradio as gr
    with pytest.raises(gr.Error):
        callbacks[0](None, "Fondo gris", "#ffffff", 12)
    demo.close()
