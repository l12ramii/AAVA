import argparse
import csv
import json
from pathlib import Path
import numpy as np

from componentes import nms, emparejar, metricas
from soporte import BASE, cargar

def main():
    parser = argparse.ArgumentParser(description="Grid Search para optimizar hiperparámetros de detección.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Conjunto de datos (por defecto: val)")
    parser.add_argument("--entrada", type=Path, default=BASE / "datos/candidatos_base.json", help="Ruta al JSON de candidatos")
    parser.add_argument("--salida", type=Path, default=BASE / "grid_search_resultados.csv", help="Ruta del CSV de salida")
    parser.add_argument("--min-recall", type=float, default=0.5, help="Umbral mínimo de Recall (por defecto: 0.5)")
    parser.add_argument("--min-precision", type=float, default=0.3, help="Umbral mínimo de Precisión (por defecto: 0.3)")
    args = parser.parse_args()

    split = args.split
    entrada = args.entrada
    salida_csv = args.salida
    min_recall = args.min_recall
    min_precision = args.min_precision

    cache = json.loads(entrada.read_text(encoding="utf-8"))
    registros = [r for r in cargar() if r["split"] == split]
    n_gt_total = sum(len(r["boxes"]) for r in registros)

    # Pre-cargar predicciones y anotaciones GT en memoria para máxima velocidad
    datos_imagenes = []
    for r in registros:
        img_id = r["id"]
        registro_pred = cache["predicciones"][img_id]
        boxes = np.asarray(registro_pred["boxes"], dtype=float).reshape(-1, 4)
        scores = np.asarray(registro_pred["scores"], dtype=float)
        gt = r["boxes"]
        datos_imagenes.append({
            "id": img_id,
            "boxes": boxes,
            "scores": scores,
            "gt": gt
        })

    # Rangos de búsqueda para los hiperparámetros
    # (El score mínimo en la caché es -1.0)
    scores_grid = np.round(np.arange(-0.8, 1.05, 0.05), 3)
    nms_grid = np.round(np.arange(0.1, 0.75, 0.05), 3)
    iou_eval_grid = np.round(np.arange(0.3, 0.75, 0.05), 3)

    resultados = []

    print(f"Iniciando Grid Search sobre '{split}'...")
    print(f"Combinaciones a evaluar: {len(scores_grid) * len(nms_grid) * len(iou_eval_grid)}")

    for score_umbral in scores_grid:
        for nms_umbral in nms_grid:
            for iou_eval in iou_eval_grid:
                todos_tp = []
                todos_fp = []

                for item in datos_imagenes:
                    b_orig = item["boxes"]
                    s_orig = item["scores"]
                    gt = item["gt"]

                    # 1. Filtrar por score
                    mascara = s_orig >= score_umbral
                    b = b_orig[mascara]
                    s = s_orig[mascara]

                    # 2. Aplicar NMS
                    indices = nms(b, s, umbral=nms_umbral)
                    b = b[indices]
                    s = s[indices]

                    # 3. Emparejar con GT
                    tp, fp = emparejar(b, s, gt, umbral=iou_eval)
                    todos_tp.extend(tp.tolist())
                    todos_fp.extend(fp.tolist())

                # 4. Calcular métricas globales
                m = metricas(todos_tp, todos_fp, n_gt_total)
                prec = m["precision"]
                rec = m["recall"]

                # Condición: recall > min_recall y precision > min_precision
                if rec > min_recall and prec > min_precision:
                    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    resultados.append({
                        "score": score_umbral,
                        "nms": nms_umbral,
                        "iou_eval": iou_eval,
                        "TP": m["TP"],
                        "FP": m["FP"],
                        "FN": m["FN"],
                        "precision": round(prec, 4),
                        "recall": round(rec, 4),
                        "f1_score": round(f1, 4)
                    })

    # Ordenar resultados por F1-score descendente y luego por precisión
    resultados.sort(key=lambda x: (x["f1_score"], x["precision"], x["recall"]), reverse=True)

    # Guardar en CSV
    columnas = ["score", "nms", "iou_eval", "TP", "FP", "FN", "precision", "recall", "f1_score"]
    with open(salida_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n¡Búsqueda finalizada!")
    print(f"Configuraciones que cumplen (recall > {min_recall} y precision > {min_precision}): {len(resultados)}")
    print(f"Archivo guardado en: {salida_csv.resolve()}\n")

    if resultados:
        print("--- Top 5 Mejores Configuraciones (por F1-score) ---")
        for i, res in enumerate(resultados[:5], 1):
            print(f"#{i}: Score={res['score']}, NMS={res['nms']}, IoU_eval={res['iou_eval']} | "
                  f"TP={res['TP']}, FP={res['FP']}, FN={res['FN']} | "
                  f"Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, F1={res['f1_score']:.4f}")
    else:
        print("Ninguna combinación superó simultáneamente ambos umbrales.")

if __name__ == "__main__":
    main()
