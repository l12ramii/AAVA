"""Ejecutar P1 o P2; no requiere GUI, internet ni entrenamiento."""
# MJMJ/UCO

import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from componentes import detectar, nms, emparejar, metricas
from soporte import BASE, Clasificador, cargar, dibujar

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sesion', choices=['p1','p2'])
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--escalas', type=float, nargs='+', default=[1.,.8,.64,.512])
    p.add_argument('--score', type=float, default=0.)
    p.add_argument('--nms', type=float, default=.3)
    p.add_argument('--iou-eval', type=float, default=.5)
    p.add_argument('--entrada', type=Path, help='JSON de P1; P2 usa caché suministrada si se omite')
    p.add_argument('--salida', type=Path, default=BASE/'resultados')
    a=p.parse_args()
    if a.stride<=0 or any(s<=0 for s in a.escalas): p.error('Stride y escalas positivos')
    if not 0<=a.nms<=1 or not 0<=a.iou_eval<=1: p.error('IoU debe pertenecer a [0,1]')
    a.salida.mkdir(parents=True, exist_ok=True)
    registros=[r for r in cargar() if r['split']==a.split]
    clasificador=Clasificador() if a.sesion=='p1' else None
    cache=None
    if a.sesion=='p2':
        entrada=a.entrada or BASE/'datos/candidatos_base.json'
        cache=json.loads(entrada.read_text(encoding='utf-8'))
        if a.score < cache['config']['score_min']:
            p.error('El score solicitado es inferior al mínimo guardado en el JSON')
        print('Origen de candidatos:', entrada)
    pred, todos_tp, todos_fp, resumen_imagenes = {}, [], [], []
    inicio=time.perf_counter()
    for r in registros:
        im=cv2.imread(str(BASE/'datos'/r['archivo']),cv2.IMREAD_GRAYSCALE)
        if im is None: raise FileNotFoundError(r['archivo'])
        if a.sesion=='p1':
            b,s=detectar(im,clasificador,a.escalas,a.stride,a.score)
        else:
            if r['id'] not in cache['predicciones']:
                raise ValueError('Falta imagen en JSON: '+r['id'])
            registro=cache['predicciones'][r['id']]
            b=np.asarray(registro['boxes'],dtype=float).reshape(-1,4)
            s=np.asarray(registro['scores'],dtype=float)
            b,s=b[s>=a.score],s[s>=a.score]
            antes=len(s)
            indices=nms(b,s,a.nms)
            b,s=b[indices],s[indices]
            tp,fp=emparejar(b,s,r['boxes'],a.iou_eval)
            todos_tp.extend(tp.tolist()); todos_fp.extend(fp.tolist())
            resumen_imagenes.append(dict(id=r['id'],antes_nms=antes,despues_nms=len(s),TP=int(sum(tp)),FP=int(sum(fp))))
        pred[r['id']]=dict(boxes=b.tolist(),scores=s.tolist())
        dibujar(im,b,s,r['boxes'],a.salida/(r['id']+'.png'))
    config=dict(score_min=a.score,split=a.split)
    if a.sesion=='p1': config.update(stride=a.stride,escalas=a.escalas)
    else: config.update(nms=a.nms,iou_eval=a.iou_eval,origen_config=cache['config'])
    (a.salida/'predicciones.json').write_text(json.dumps(dict(config=config,predicciones=pred),indent=2),encoding='utf-8')
    resumen=dict(sesion=a.sesion,config=config,segundos=time.perf_counter()-inicio,
                 imagenes=len(registros),cajas=sum(len(x['scores']) for x in pred.values()))
    if a.sesion=='p2':
        resumen.update(metricas(todos_tp,todos_fp,sum(len(r['boxes']) for r in registros)))
        resumen['por_imagen']=resumen_imagenes
    (a.salida/'resumen.json').write_text(json.dumps(resumen,indent=2),encoding='utf-8')
    print(json.dumps(resumen,indent=2))

if __name__=='__main__': main()
