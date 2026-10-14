"""T7: el alumno sustituye la referencia por su política justificada antes de entrenar."""


def should_stop(history, settings):
    """Contrato: history=list de épocas completas (epoch, train, val, seconds).
    settings=dict JSON de stopping.json, con description no vacía.
    Devuelve (bool, str): continuar -> (False, ""); parar -> (True, motivo no vacío).
    Función pura: no modifica argumentos, no usa test, azar, modelo ni reloj.
    Debe ser determinista, independiente de la arquitectura y usar solo el historial
    train/val. Mismo código y settings para ambos modelos; épocas finales pueden diferir.
    Los parámetros adicionales los define/documenta el alumno. Vacío -> continuar.
    """
    # TODO T7_STOP: Propón e implementa tu criterio determinista de parada; consulta CONTRATOS.md.
    raise NotImplementedError("T7_STOP: Propón e implementa tu criterio determinista de parada; consulta CONTRATOS.md.")
