"""
grafica_documento.py
--------------------
Vuelve a dibujar la gráfica de precisión frente a latencia a partir de
results/metricas_modelos.csv, con las etiquetas separadas para que se lean
bien en el documento. Usa los mismos datos que la gráfica del cuaderno.

Uso:
    python src/grafica_documento.py
Salida:
    results/precision_vs_latencia_documento.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORES = {"BART-large (inglés)": "tab:blue", "mDeBERTa-v3 (multilingüe)": "tab:orange",
           "MiniLMv2 (multilingüe)": "tab:green"}
# Desplazamiento de cada etiqueta (en puntos) para que no se encimen.
DESPLAZAMIENTO = {
    ("BART-large (inglés)", "con fine-tuning"): (24, -6),
    ("mDeBERTa-v3 (multilingüe)", "con fine-tuning"): (-150, 16),
    ("MiniLMv2 (multilingüe)", "con fine-tuning"): (14, -6),
    ("BART-large (inglés)", "sin ajuste"): (22, -6),
    ("mDeBERTa-v3 (multilingüe)", "sin ajuste"): (-150, -6),
    ("MiniLMv2 (multilingüe)", "sin ajuste"): (14, -6),
}


def main():
    tabla = pd.read_csv("results/metricas_modelos.csv")
    fig, eje = plt.subplots(figsize=(9, 5.2))
    for _, fila in tabla.iterrows():
        ajustado = fila["etapa"] == "con fine-tuning"
        eje.scatter(fila["latencia_mediana_ms"], fila["f1"], s=fila["parametros_millones"] * 3,
                    color=COLORES[fila["modelo"]], alpha=0.85 if ajustado else 0.35,
                    marker="o" if ajustado else "s", edgecolor="black", linewidth=0.6)
        eje.annotate(f"{fila['modelo']}\n{fila['etapa']} ({fila['parametros_millones']:.0f} M)",
                     (fila["latencia_mediana_ms"], fila["f1"]), textcoords="offset points",
                     xytext=DESPLAZAMIENTO[(fila["modelo"], fila["etapa"])], fontsize=8.5)
    for modelo, color in COLORES.items():
        antes = tabla[(tabla["modelo"] == modelo) & (tabla["etapa"] == "sin ajuste")].iloc[0]
        despues = tabla[(tabla["modelo"] == modelo) & (tabla["etapa"] == "con fine-tuning")].iloc[0]
        eje.annotate("", xy=(despues["latencia_mediana_ms"], despues["f1"]),
                     xytext=(antes["latencia_mediana_ms"], antes["f1"]),
                     arrowprops=dict(arrowstyle="->", color=color, alpha=0.6))
    eje.axhline(0.80, color="gray", linestyle=":", label="Umbral de F1 = 0.80")
    eje.axvline(100, color="gray", linestyle="--", label="Umbral de latencia = 100 ms")
    eje.set_xlabel("Latencia mediana por mensaje (ms)")
    eje.set_ylabel("F1-score ponderado (validación)")
    eje.set_xlim(-22, 112)
    eje.set_ylim(0.25, 1.12)
    eje.grid(linestyle="--", alpha=0.5)
    eje.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("results/precision_vs_latencia_documento.png", dpi=150)
    print("Gráfica guardada en results/precision_vs_latencia_documento.png")


if __name__ == "__main__":
    main()
