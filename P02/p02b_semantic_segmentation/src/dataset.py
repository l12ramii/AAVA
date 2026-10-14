"""Contratos T0: carga, normalización y transformaciones conjuntas. Etiquetas de salida: fondo=0, mascota=1, ignorar=255."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF

IGNORE_INDEX = 255
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def trimap_tensor(trimap):
    """T0_MAP: PIL trimap -> long HxW; 1->1, 2->0, 3->255; rechaza otras etiquetas."""
    # TODO T0_MAP: Implementa trimap_tensor según su contrato; etiquetas inválidas -> ValueError.
    raise NotImplementedError("T0_MAP: Implementa trimap_tensor según su contrato; etiquetas inválidas -> ValueError.")


def image_tensor(image, size):
    """T0_IMAGE: PIL -> float32 3xsize xsize; RGB, bilinear, [0,1], MEAN/STD."""
    # TODO T0_IMAGE: Implementa imagen RGB normalizada float32 según el contrato.
    raise NotImplementedError("T0_IMAGE: Implementa imagen RGB normalizada float32 según el contrato.")


def paired_transform(image, trimap, size=128, flip=False):
    """T0_PAIR: flip determinista conjunto si flip=True; imagen bilinear, trimap nearest.
    Devuelve (imagen float32 3xsize xsize, etiqueta long size xsize).
    La decisión aleatoria corresponde al dataset, nunca a esta función.
    """
    # TODO T0_PAIR: Implementa transformaciones conjuntas y resize según el contrato.
    raise NotImplementedError("T0_PAIR: Implementa transformaciones conjuntas y resize según el contrato.")


def load_pair(base, name):
    """T0_LOAD: base=.../oxford-iiit-pet, ID sin extensión -> PIL RGB y PIL trimap.
    Leer images/<ID>.jpg y annotations/trimaps/<ID>.png, cerrar ambos archivos,
    devolver copias independientes a tamaño original. No aplicar EXIF ni resize:
    las anotaciones oficiales están alineadas con los píxeles almacenados.
    """
    # TODO T0_LOAD: Lee y devuelve la pareja PIL original independiente de archivos cerrados.
    raise NotImplementedError("T0_LOAD: Lee y devuelve la pareja PIL original independiente de archivos cerrados.")


def make_manifest(root, output, counts=(1500, 300, 300), seed=2026, download=False):
    """Selección fija por SHA256 de identificadores; nunca modifica los datos originales."""
    root, output = Path(root), Path(output)
    if len(counts) != 3 or any(n <= 0 for n in counts):
        raise ValueError("Se requieren tres tamaños positivos.")
    OxfordIIITPet(root, split="trainval", target_types="segmentation", download=download)
    ann = root / "oxford-iiit-pet" / "annotations"
    def ids(split):
        return [line.split()[0] for line in (ann / f"{split}.txt").read_text().splitlines() if line.strip()]
    tv, test = ids("trainval"), ids("test")
    if len(set(tv)) != len(tv) or len(set(test)) != len(test) or set(tv) & set(test):
        raise ValueError("Las listas oficiales contienen duplicados o solapamientos.")
    if counts[0] + counts[1] > len(tv) or counts[2] > len(test):
        raise ValueError("Subconjuntos mayores que las particiones oficiales.")
    key = lambda name: (hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(), name)
    tv, test = sorted(tv, key=key), sorted(test, key=key)
    result = {"schema": 1, "seed": seed, "counts": list(counts),
              "selection": "SHA256(seed:id), train/val from official trainval, test from official test",
              "official_sha256": {s: hashlib.sha256((ann / f"{s}.txt").read_bytes()).hexdigest()
                                  for s in ("trainval", "test")},
              "train": tv[:counts[0]], "val": tv[counts[0]:sum(counts[:2])], "test": test[:counts[2]]}
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("El manifiesto existente difiere: utilice otra ruta, no cambie la partición en curso.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def read_manifest(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = [result[s] for s in ("train", "val", "test")]
    flat = sum(groups, [])
    if any(not g for g in groups) or len(set(flat)) != len(flat):
        raise ValueError("Particiones vacías, duplicadas o solapadas.")
    if any(Path(x).name != x or "/" in x or "\\" in x for x in flat):
        raise ValueError("Identificador de imagen inválido.")
    return result


class PetDataset(Dataset):
    def __init__(self, root, manifest, split, size=128, augment=False):
        if size < 32 or split not in ("train", "val", "test"):
            raise ValueError("Tamaño >=32 y split train/val/test requeridos.")
        if augment and split != "train":
            raise ValueError("Solo se aumenta el conjunto de entrenamiento.")
        self.base = Path(root) / "oxford-iiit-pet"
        self.ids = read_manifest(manifest)[split]
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """T0_DATASET: devuelve pareja transformada; flip con p=.5 solo si augment.
        Usar torch.rand para que el estado de reanudación restaure la augmentación.
        """
        # TODO T0_DATASET: Implementa carga y augmentación solo train según el contrato.
        raise NotImplementedError("T0_DATASET: Implementa carga y augmentación solo train según el contrato.")


def make_inference_subset(manifest, output, count=30):
    """Infraestructura: subconjunto fijo de validación, sin observar predicciones."""
    record = read_manifest(manifest)
    if not 0 < count <= len(record["val"]):
        raise ValueError("Se necesitan al menos count imágenes de validación.")
    ids = sorted(record["val"], key=lambda n: (hashlib.sha256(f"infer:2026:{n}".encode()).hexdigest(), n))[:count]
    subset = {"schema": 1, "split": "val", "count": count, "ids": ids,
              "manifest_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()}
    path = Path(output)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != subset:
        raise ValueError("Subconjunto existente diferente: usa otra ruta.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    return subset


class OriginalPetDataset(Dataset):
    """Lectura T0_ORIGINAL para inferencia: sin resize ni augmentación; no usar DataLoader por lotes."""
    def __init__(self, root, manifest, subset):
        self.base = Path(root) / "oxford-iiit-pet"
        record = read_manifest(manifest)
        selection = json.loads(Path(subset).read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        self.ids = selection["ids"]
        expected = sorted(record["val"], key=lambda n: (hashlib.sha256(f"infer:2026:{n}".encode()).hexdigest(), n))[:selection["count"]]
        if (selection["split"] != "val" or selection["manifest_sha256"] != digest
                or not self.ids or len(set(self.ids)) != len(self.ids)
                or selection["count"] != len(self.ids) or self.ids != expected or not set(self.ids) <= set(record["val"])):
            raise ValueError("El subconjunto no corresponde a esta validación.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """T0_ORIGINAL: devuelve (PIL RGB original, long HxW remapeado); conserva 255."""
        # TODO T0_ORIGINAL: Carga imagen y target originales para evaluar inferencia.
        raise NotImplementedError("T0_ORIGINAL: Carga imagen y target originales para evaluar inferencia.")
