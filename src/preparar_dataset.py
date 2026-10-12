"""
preparar_dataset.py
-------------------
Descarga de Hugging Face Hub el dataset público de mensajes de soporte al
cliente de Bitext y prepara la muestra balanceada que se usa para comparar
y ajustar los modelos del chatbot de CAB.

Fuente: Bitext. (s.f.). Bitext customer support LLM chatbot training dataset
        [Conjunto de datos]. Hugging Face.
        https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
Licencia del dataset original: CDLA-Sharing-1.0.

Pasos:
  1. Carga el dataset con la librería datasets de Hugging Face.
  2. Agrupa las 27 intenciones originales en las 5 que necesita el chatbot:
       estatus_orden        <- track_order
       fecha_entrega        <- delivery_period
       queja                <- complaint
       contactar_ejecutivo  <- contact_human_agent, contact_customer_service
       otro                 <- el resto (facturas, pagos, cancelaciones, cuentas...)
  3. Elimina mensajes repetidos.
  4. Toma una muestra balanceada (mismo número de mensajes por clase) con semilla fija.

Uso:
    python src/preparar_dataset.py
Salida:
    data/mensajes_clientes_muestra.csv
"""

import hashlib
import os

import pandas as pd
from datasets import load_dataset

DATASET_HF = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RUTA_SALIDA = os.path.join("data", "mensajes_clientes_muestra.csv")
SEMILLA = 42
MENSAJES_POR_CLASE = 400

# Intenciones del chatbot de CAB y las intenciones originales que agrupa cada una.
INTENCIONES_CAB = {
    "estatus_orden": ["track_order"],
    "fecha_entrega": ["delivery_period"],
    "queja": ["complaint"],
    "contactar_ejecutivo": ["contact_human_agent", "contact_customer_service"],
}


def asignar_intencion(intent_original):
    """Convierte la intención original del dataset en una de las 5 clases del chatbot."""
    for clase, originales in INTENCIONES_CAB.items():
        if intent_original in originales:
            return clase
    return "otro"


def preparar(tabla):
    """Limpia el dataset original y regresa la muestra balanceada."""
    faltantes = [i for lista in INTENCIONES_CAB.values() for i in lista
                 if i not in set(tabla["intent"])]
    if faltantes:
        raise ValueError(f"El dataset no contiene las intenciones esperadas: {faltantes}")

    tabla = tabla.dropna(subset=["instruction", "intent"]).copy()
    tabla["mensaje"] = (tabla["instruction"].astype(str)
                        .str.replace(r"\s+", " ", regex=True).str.strip())
    tabla = tabla[tabla["mensaje"].str.len() > 0]
    tabla = tabla.drop_duplicates(subset="mensaje")
    tabla["intencion"] = tabla["intent"].apply(asignar_intencion)
    tabla["num_palabras"] = tabla["mensaje"].str.split().str.len()
    # Se ordena antes de muestrear para que el resultado no dependa del orden del archivo.
    tabla = tabla.sort_values(["intencion", "mensaje"]).reset_index(drop=True)
    disponibles = tabla["intencion"].value_counts()
    print("Mensajes disponibles por clase después de la limpieza:")
    print(disponibles.to_string())

    por_clase = min(MENSAJES_POR_CLASE, int(disponibles.min()))
    muestra = pd.concat([
        grupo.sample(n=por_clase, random_state=SEMILLA)
        for _, grupo in tabla.groupby("intencion", sort=True)
    ])
    muestra = muestra.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    muestra.insert(0, "id", range(1, len(muestra) + 1))
    muestra = muestra.rename(columns={"intent": "intencion_original", "category": "categoria_original"})
    return muestra[["id", "mensaje", "intencion", "intencion_original", "categoria_original", "num_palabras"]]


def main():
    original = load_dataset(DATASET_HF, split="train").to_pandas()
    print(f"Dataset descargado de Hugging Face Hub: {DATASET_HF}")
    print(f"Mensajes en el dataset original: {len(original)}")
    print(f"Columnas originales: {list(original.columns)}")
    print(f"Intenciones originales: {original['intent'].nunique()}")
    muestra = preparar(original)
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    muestra.to_csv(RUTA_SALIDA, index=False, encoding="utf-8", lineterminator="\n")
    with open(RUTA_SALIDA, "rb") as f:
        huella = hashlib.md5(f.read()).hexdigest()
    print(f"\nMuestra guardada: {RUTA_SALIDA} ({len(muestra)} mensajes, "
          f"{len(muestra) // muestra['intencion'].nunique()} por clase)")
    print(f"Huella MD5 del archivo: {huella}")


if __name__ == "__main__":
    main()
