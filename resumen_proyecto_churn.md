# 📋 Bitácora del Proyecto — Predicción de Churn en Telecomunicaciones
**Big Data DD283 | Universidad Autónoma del Perú | 2026-1**

> Este documento resume todo lo avanzado hasta ahora. Si pierdes el historial de chat, puedes pegar este archivo completo al inicio de una conversación nueva para que el asistente recupere el contexto.

---

## 1. Contexto del proyecto
- **Tema:** predecir con 30 días de anticipación qué clientes postpago/control van a cancelar su servicio (Churn) en el mercado de telecom peruano (Claro, Movistar, Entel).
- **Meta del modelo:** Recall > 80% en la detección de Churn.
- **Stack declarado en el README:** Python + PySpark (ETL y MLlib), Hadoop HDFS/Parquet, MongoDB Atlas, Power BI.
- **Dataset actualmente en uso:** el dataset real de IBM (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7,043 clientes). **Pendiente:** el README promete un dataset sintético de 500,000 clientes peruanos generado con Faker (con ARPU, zona geográfica, etc.) — todavía no se ha construido ese generador.

## 2. Estado de cada archivo

| Archivo | Estado | Notas clave |
|---|---|---|
| `README.md` | ✅ Revisado y mejorado | Se agregó sección 📊 Resultados (con métricas objetivo) y referencia a `dssg/telco-churn`. |
| `01_exploracion_EDA.ipynb` | ✅ OK, no se modificó | Decisión correcta: imputar `TotalCharges` nulo con **0.0** (clientes nuevos con `tenure=0`, aún no facturados). |
| `02_pipeline_procesamiento.ipynb` | ⚠️ Pendiente de un ajuste | Lógicamente sólido (ingesta, limpieza, indexación, ensamblado). **Pendiente:** todavía imputa con la mediana en vez de 0.0 — hay que alinearlo con el EDA (no se ha hecho aún). |
| `03_modelo_ml.ipynb` | ✅ Reconstruido y corregido | Ver detalle abajo. |
| `04_dashboard.ipynb` | ✅ Reconstruido y corregido | Ver detalle abajo. |
| `src/pipeline.py`, `src/modelo.py` | ❌ No creados todavía | Mencionados en la estructura del repo pero no se han generado. |

## 3. Qué se hizo en `03_modelo_ml.ipynb`
- Limpieza de `TotalCharges` alineada con el EDA (imputación con `0.0`, no mediana).
- Se conserva `customerID` en todo el flujo (antes se perdía en el `VectorAssembler`).
- Se entrenan **2 modelos**: Regresión Logística (baseline) y Random Forest (para cumplir lo que promete el README de RF/GBT).
- Evaluación completa: ROC AUC, Accuracy, **Precision/Recall/F1 sobre la clase Churn=1** (antes solo se medía ROC AUC, que no basta con clases desbalanceadas).
- **Threshold tuning**: se prueba un umbral más bajo (0.35) para subir el Recall, ya que el dataset está desbalanceado (~27% Churn).
- Se calcula la **importancia de variables** del Random Forest.
- Se selecciona automáticamente el modelo con mayor Recall y se exportan los resultados.
- **Bugs corregidos en este notebook:**
  1. Un `UDF` de Python sobre la columna `probability` (tipo `Vector` de Spark) causaba `PickleException: ... DenseVector`. Se reemplazó por `vector_to_array` (función nativa de Spark).
  2. `predictions.write.csv(...)` fallaba en Windows con el error de `winutils.exe` / `HADOOP_HOME` no configurado. Se reemplazó por `.toPandas().to_csv(...)`.
- **Archivos que exporta (en `data/sample/`):**
  - `predicciones_churn_dashboard.csv` (customerID, churn_real, churn_predicho, probabilidad_churn)
  - `importancia_variables_churn.csv`

## 4. Qué se hizo en `04_dashboard.ipynb`
- Carga las predicciones reales del notebook 3 + enriquece con `Contract`, `MonthlyCharges`, `tenure` del CSV original (join por `customerID`).
- KPIs ejecutivos (Recall, Precision, F1, Accuracy) + matriz de confusión visual.
- Distribución de probabilidad de Churn (diagnóstico del modelo).
- **Dashboard ejecutivo 2x2** con impacto de negocio: clientes por banda de riesgo (Alto ≥70%, Medio 40-70%, Bajo <40%), **MRR (ingreso mensual recurrente) en riesgo**, boxplot de probabilidad por tipo de contrato, y mapa de vulnerabilidad (tenure vs cargo mensual). Se exporta como imagen `dashboard_ejecutivo_fase4.png`.
- Top 10 factores que más explican el Churn (gráfico de barras).
- Lista priorizada de clientes de alto riesgo a contactar, exportada a `lista_retencion_prioritaria.csv`.
- **Importante:** se descartó una versión de este notebook hecha por un compañero porque simulaba `Probability_Churn` con `np.random` + una regla manual — nunca usaba el modelo real. Hay que avisarle para que no suba esa versión a la entrega final.

## 5. Decisiones tomadas (para no repetir la discusión)
- Imputación de `TotalCharges`: **0.0**, no mediana (lógica de negocio: clientes nuevos sin facturación).
- Segmentos de riesgo: Alto ≥0.70, Medio 0.40–0.70, Bajo <0.40.
- Se compara LR vs RF; se elige el de mayor Recall automáticamente en el notebook 3.
- No se usó SMOTE ni XGBoost (rompería la consistencia con el stack PySpark/Big Data declarado en el README); en su lugar se usó *threshold tuning*. (Pendiente evaluar `weightCol` como alternativa nativa de Spark si el Recall sigue bajo.)

## 6. Pendientes / Qué sigue
1. **Correr los notebooks 2 → 3 → 4 de punta a punta** con las correcciones, para tener números reales de Recall/Precision y confirmar si se cumple la meta (>80%).
2. **Alinear el notebook 2** con la imputación de 0.0 (actualmente sigue usando mediana).
3. **Actualizar la sección 📊 Resultados del README** con los números reales obtenidos.
4. Decidir si se necesita mejorar el Recall con `weightCol` (class weighting nativo de Spark) si el threshold tuning no es suficiente.
5. Construir el generador de datos sintéticos peruanos (Faker, 500k clientes) si el profesor lo exige para la siguiente entrega — actualmente todo corre sobre el dataset real de IBM (7,043 filas).
6. Crear `src/pipeline.py` y `src/modelo.py` (mencionados en la estructura del repo, aún no existen).
7. Conectar los CSV exportados (`predicciones_churn_dashboard.csv`, `importancia_variables_churn.csv`, `lista_retencion_prioritaria.csv`, `dashboard_ejecutivo_fase4.png`) a Power BI para el dashboard interactivo final.
8. **Pendiente del usuario:** compartir la rúbrica del profesor para verificar que todo lo avanzado cumple lo que se va a calificar.
