"""Descarga opcional y manifiestos fijos. Ejecutar antes de entrenar."""
import argparse
from src.dataset import make_manifest, make_inference_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="splits/default.json")
    parser.add_argument("--counts", nargs=3, type=int, default=[1500, 300, 300], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--inference-subset", help="JSON de 30 imágenes fijas de validación")
    args = parser.parse_args()
    result = make_manifest(args.data, args.output, args.counts, args.seed, args.download)
    if args.inference_subset:
        make_inference_subset(args.output, args.inference_subset)
    print({split: len(result[split]) for split in ("train", "val", "test")})


if __name__ == "__main__":
    main()
