"""
analisis_complementario.py
--------------------------
Análisis adicional a partir de las predicciones guardadas por el cuaderno
(results/predicciones_validacion.csv y results/predicciones_test.csv).
No necesita GPU ni descargar modelos.

Calcula dos cosas sobre el modelo seleccionado, antes y después del ajuste:
  1. Errores de ruta del chatbot. El chatbot responde solo las intenciones
     estatus_orden y fecha_entrega; las demás (queja, contactar_ejecutivo y
     otro) se pasan a un ejecutivo. Se cuenta cuántos mensajes que debían
     llegar a un ejecutivo se habrían respondido en automático, y al revés.
  2. El accuracy según la confianza de la predicción, que sirve para proponer
     un umbral de revisión humana.

Uso:
    python src/analisis_complementario.py
Salida:
    results/errores_de_ruta_validacion.csv   (los tres modelos, validación)
    results/errores_de_ruta.csv              (modelo seleccionado, prueba)
    results/accuracy_por_confianza.csv       (modelo seleccionado, prueba)
"""

import pandas as pd

AUTOMATICAS = {"estatus_orden", "fecha_entrega"}        # las responde el chatbot
ETAPAS = {"sin ajuste": "sin_ajuste", "con fine-tuning": "con_fine_tuning"}
CORTES_CONFIANZA = [0.0, 0.60, 0.80, 0.90, 1.0001]
ETIQUETAS_CORTES = ["menor a 0.60", "0.60 a 0.80", "0.80 a 0.90", "0.90 o más"]


def ruta(intencion):
    """Indica si el chatbot respondería el mensaje o lo pasaría a un ejecutivo."""
    return "automatica" if intencion in AUTOMATICAS else "ejecutivo"


def main():
    pred = pd.read_csv("results/predicciones_test.csv")
    ruta_real = pred["intencion"].apply(ruta)

    filas_ruta, filas_confianza = [], []
    for etapa, sufijo in ETAPAS.items():
        ruta_predicha = pred[f"pred_{sufijo}"].apply(ruta)
        debian_ejecutivo = ruta_real == "ejecutivo"
        debian_automatica = ruta_real == "automatica"
        filas_ruta.append({
            "etapa": etapa,
            "mensajes": len(pred),
            "debian_ir_a_ejecutivo": int(debian_ejecutivo.sum()),
            "respondidos_en_automatico_por_error": int((debian_ejecutivo & (ruta_predicha == "automatica")).sum()),
            "debian_responderse_en_automatico": int(debian_automatica.sum()),
            "enviados_a_ejecutivo_por_error": int((debian_automatica & (ruta_predicha == "ejecutivo")).sum()),
            "ruta_correcta": int((ruta_real == ruta_predicha).sum()),
        })

        datos = pd.DataFrame({
            "acierto": pred["intencion"] == pred[f"pred_{sufijo}"],
            "rango_confianza": pd.cut(pred[f"confianza_{sufijo}"], bins=CORTES_CONFIANZA,
                                      labels=ETIQUETAS_CORTES, right=False),
        })
        resumen = (datos.groupby("rango_confianza", observed=False)["acierto"]
                   .agg(mensajes="count", aciertos="sum").reset_index())
        resumen["accuracy"] = (resumen["aciertos"] / resumen["mensajes"]).round(4)
        resumen.insert(0, "etapa", etapa)
        filas_confianza.append(resumen)

    # Lo mismo para los tres modelos en el conjunto de validación.
    val = pd.read_csv("results/predicciones_validacion.csv")
    ruta_real_val = val["intencion"].apply(ruta)
    filas_val = []
    for columna in [c for c in val.columns if c.startswith(("base_", "ajustado_"))]:
        etapa = "sin ajuste" if columna.startswith("base_") else "con fine-tuning"
        modelo = columna.split("_", 1)[1]
        ruta_predicha = val[columna].apply(ruta)
        filas_val.append({
            "modelo": modelo, "etapa": etapa, "mensajes": len(val),
            "respondidos_en_automatico_por_error": int(((ruta_real_val == "ejecutivo") & (ruta_predicha == "automatica")).sum()),
            "enviados_a_ejecutivo_por_error": int(((ruta_real_val == "automatica") & (ruta_predicha == "ejecutivo")).sum()),
            "ruta_correcta": int((ruta_real_val == ruta_predicha).sum()),
        })
    errores_val = pd.DataFrame(filas_val)
    errores_val.to_csv("results/errores_de_ruta_validacion.csv", index=False)
    print("ERRORES DE RUTA DEL CHATBOT POR MODELO (conjunto de validación)")
    print(errores_val.to_string(index=False))
    print()

    errores = pd.DataFrame(filas_ruta)
    errores.to_csv("results/errores_de_ruta.csv", index=False)
    print("ERRORES DE RUTA DEL CHATBOT (conjunto de prueba)")
    print(errores.to_string(index=False))

    confianza = pd.concat(filas_confianza, ignore_index=True)
    confianza.to_csv("results/accuracy_por_confianza.csv", index=False)
    print("\nACCURACY SEGÚN LA CONFIANZA DE LA PREDICCIÓN (conjunto de prueba)")
    print(confianza.to_string(index=False))


if __name__ == "__main__":
    main()
