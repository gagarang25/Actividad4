"""
funciones_evaluacion.py
-----------------------
Funciones de evaluación que usa el cuaderno de Colab, reunidas en un módulo
para poder reutilizarlas en otros scripts (por ejemplo, en la Fase 2 del
proyecto). Son las mismas que aparecen en la sección 7 del cuaderno.

Ejemplo de uso:
    import pandas as pd
    from funciones_evaluacion import MODELOS, evaluar_modelo

    df = pd.read_csv("data/mensajes_clientes_muestra.csv")
    resultado, predichas, confianzas = evaluar_modelo(
        "mDeBERTa-v3 (multilingüe)", MODELOS["mDeBERTa-v3 (multilingüe)"], "sin ajuste",
        df["mensaje"].tolist(), df["intencion"].tolist(), dispositivo=0,
    )
"""

import gc
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import pipeline

# Identificadores de los tres modelos en Hugging Face Hub.
MODELOS = {
    "BART-large (inglés)": "facebook/bart-large-mnli",
    "mDeBERTa-v3 (multilingüe)": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "MiniLMv2 (multilingüe)": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
}

CLASES = ["estatus_orden", "fecha_entrega", "queja", "contactar_ejecutivo", "otro"]

# Descripción en inglés de cada clase. Los modelos sin ajuste eligen la frase
# que mejor describe el mensaje (clasificación zero-shot).
DESCRIPCIONES = {
    "estatus_orden": "the status or tracking of an order",
    "fecha_entrega": "how long the delivery takes or when the order will arrive",
    "queja": "making a complaint",
    "contactar_ejecutivo": "talking to a human agent or contacting customer service",
    "otro": "another topic such as invoices, payments, refunds, cancellations or accounts",
}
PLANTILLA = "This message is about {}."
FRASE_A_CLASE = {frase: clase for clase, frase in DESCRIPCIONES.items()}

TAMANO_LOTE = 16      # mensajes que se procesan juntos en la prueba por lotes
N_LATENCIA = 100      # mensajes que se envían de uno en uno para medir latencia
N_REPETICIONES = 3    # veces que se repite la medición de latencia


def calcular_metricas(reales, predichas):
    """Calcula accuracy, precision, recall y F1 (promedio ponderado) y F1 macro."""
    accuracy = accuracy_score(reales, predichas)
    precision, recall, f1, _ = precision_recall_fscore_support(
        reales, predichas, average="weighted", zero_division=0
    )
    f1_macro = f1_score(reales, predichas, average="macro", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }


def sincronizar():
    """Espera a que la GPU termine su trabajo para que el tiempo medido sea real."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def como_lista(salida):
    """Los pipelines regresan un diccionario si reciben un solo texto; aquí se unifica."""
    return salida if isinstance(salida, list) else [salida]


def predecir_zero_shot(clasificador, textos):
    """Clasifica con un modelo sin ajuste: elige la descripción con mayor puntaje."""
    salidas = como_lista(clasificador(
        textos, candidate_labels=list(DESCRIPCIONES.values()),
        hypothesis_template=PLANTILLA, batch_size=TAMANO_LOTE,
    ))
    return [(FRASE_A_CLASE[s["labels"][0]], float(s["scores"][0])) for s in salidas]


def predecir_ajustado(clasificador, textos):
    """Clasifica con un modelo ya ajustado, que responde directamente con la clase."""
    salidas = como_lista(clasificador(textos, batch_size=TAMANO_LOTE, truncation=True))
    return [(s["label"], float(s["score"])) for s in salidas]


def medir_clasificador(nombre, id_modelo, etapa, clasificador, predecir, textos, etiquetas_reales):
    """Clasifica los textos con un pipeline ya cargado y mide precisión y latencia."""
    # 1. Calentamiento: la primera inferencia siempre es más lenta y no se mide.
    predecir(clasificador, textos[:8])
    sincronizar()

    # 2. Inferencia por lotes sobre todos los textos (de aquí salen las métricas).
    inicio = time.perf_counter()
    salidas = predecir(clasificador, textos)
    sincronizar()
    tiempo_lote = time.perf_counter() - inicio

    # 3. Latencia: un mensaje a la vez, como llegaría al chatbot.
    #    Se repite varias veces porque en una GPU compartida el tiempo varía.
    tiempos_ms = []
    for _ in range(N_REPETICIONES):
        for texto in textos[:N_LATENCIA]:
            inicio = time.perf_counter()
            predecir(clasificador, texto)
            sincronizar()
            tiempos_ms.append((time.perf_counter() - inicio) * 1000)

    # 4. Métricas de clasificación.
    predichas = [clase for clase, _ in salidas]
    confianzas = [confianza for _, confianza in salidas]
    parametros = sum(p.numel() for p in clasificador.model.parameters())
    resultado = {"modelo": nombre, "id_modelo": id_modelo, "etapa": etapa}
    resultado.update(calcular_metricas(etiquetas_reales, predichas))
    resultado.update({
        "latencia_mediana_ms": float(np.median(tiempos_ms)),
        "latencia_media_ms": float(np.mean(tiempos_ms)),
        "latencia_p95_ms": float(np.percentile(tiempos_ms, 95)),
        "mensajes_por_segundo": len(textos) / tiempo_lote,
        "parametros_millones": parametros / 1e6,
        "memoria_gpu_pico_mb": (torch.cuda.max_memory_allocated() / 1024**2
                                if torch.cuda.is_available() else 0.0),
    })
    return resultado, predichas, confianzas


def evaluar_modelo(nombre, origen, etapa, textos, etiquetas_reales, dispositivo, id_modelo=None):
    """Carga un modelo, lo evalúa y libera la memoria al terminar.

    etapa = "sin ajuste": modelo del Hub usado con clasificación zero-shot.
    etapa = "con fine-tuning": modelo ajustado y guardado en una carpeta local.
    """
    hay_gpu = torch.cuda.is_available() and dispositivo != -1
    if hay_gpu:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if etapa == "sin ajuste":
        clasificador = pipeline("zero-shot-classification", model=origen, device=dispositivo)
        predecir = predecir_zero_shot
    else:
        clasificador = pipeline("text-classification", model=origen, device=dispositivo)
        predecir = predecir_ajustado

    resultado, predichas, confianzas = medir_clasificador(
        nombre, id_modelo or origen, etapa, clasificador, predecir, textos, etiquetas_reales
    )

    # Se libera la memoria antes de cargar el siguiente modelo.
    del clasificador
    gc.collect()
    if hay_gpu:
        torch.cuda.empty_cache()
    return resultado, predichas, confianzas
