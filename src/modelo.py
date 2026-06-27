import os
import argparse

import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# Reutilizamos las funciones del pipeline de procesamiento (no se duplica código).
from pipeline import (
    crear_sesion_spark,
    ingesta,
    limpiar,
    indexar_categoricas,
    ensamblar_features,
    COLUMNAS_FEATURES,
)

load_dotenv()

PREDICCIONES_OUTPUT = "data/sample/predicciones_churn_dashboard.csv"
IMPORTANCIAS_OUTPUT = "data/sample/importancia_variables_churn.csv"
MONGO_DB = "telco_churn_db"
MONGO_COLLECTION = "perfiles_riesgo_clientes"


def evaluar_modelo(predictions, nombre_modelo: str) -> dict:
    """Calcula ROC AUC, Accuracy, Precision/Recall/F1 sobre la clase Churn=1."""
    print(f"\n{'=' * 55}")
    print(f"EVALUACIÓN: {nombre_modelo}")
    print(f"{'=' * 55}")

    evaluator_auc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC"
    )
    roc_auc = evaluator_auc.evaluate(predictions)

    evaluator_mc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    recall = evaluator_mc.evaluate(predictions, {evaluator_mc.metricName: "recallByLabel", evaluator_mc.metricLabel: 1.0})
    precision = evaluator_mc.evaluate(predictions, {evaluator_mc.metricName: "precisionByLabel", evaluator_mc.metricLabel: 1.0})
    f1 = evaluator_mc.evaluate(predictions, {evaluator_mc.metricName: "fMeasureByLabel", evaluator_mc.metricLabel: 1.0})
    accuracy = evaluator_mc.evaluate(predictions, {evaluator_mc.metricName: "accuracy"})

    print(f"ROC AUC:               {roc_auc:.4f}")
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Precision (Churn=1):   {precision:.4f}")
    print(f"Recall (Churn=1):      {recall:.4f}   <-- meta del proyecto: > 0.80")
    print(f"F1-Score (Churn=1):    {f1:.4f}")

    cumple_meta = "SI" if recall > 0.80 else "NO"
    print(f"¿Cumple Recall > 80%?  {cumple_meta}")

    return {"modelo": nombre_modelo, "roc_auc": roc_auc, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1}


def segmentar_riesgo(p: float) -> str:
    if p >= 0.70:
        return "Alto"
    elif p >= 0.40:
        return "Medio"
    return "Bajo"


def guardar_en_mongo(registros: list, coleccion: str = MONGO_COLLECTION):
    """
    Guarda el perfil de riesgo de cada cliente en MongoDB.
    Si no hay MONGO_URI en el .env, se omite sin romper el script.
    """
    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        print("[AVISO] MONGO_URI no está configurado en el .env. Se omite el guardado en MongoDB.")
        print("        Copia .env.example a .env y agrega tu connection string.")
        return

    try:
        from pymongo import MongoClient

        cliente = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        cliente.admin.command("ping")  # valida que la conexión funcione

        db = cliente[MONGO_DB]
        db[coleccion].delete_many({})  # modo "overwrite" simple
        if registros:
            db[coleccion].insert_many(registros)
        print(f"[OK] {len(registros)} perfiles de riesgo guardados en MongoDB "
              f"(db='{MONGO_DB}', colección='{coleccion}')")
    except Exception as e:
        print(f"[ERROR] No se pudo guardar en MongoDB: {e}")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación del modelo de Churn")
    parser.add_argument("--umbral", type=float, default=0.35,
                        help="Umbral de decisión a probar para subir el Recall")
    args = parser.parse_args()

    spark = crear_sesion_spark()

    # --- 1. Pipeline de datos (reutilizando pipeline.py) ---
    df = ingesta(spark)
    df = limpiar(df)
    df = indexar_categoricas(df)

    # Conservamos variables de negocio para enriquecer el perfil final (no entran al modelo)
    df_negocio = df.select("customerID", "Contract", "MonthlyCharges", "tenure")

    df_vectorizado = ensamblar_features(df)
    train_data, test_data = df_vectorizado.randomSplit([0.7, 0.3], seed=42)

    # --- Balanceo de clases (weightCol) ---
    # El dataset tiene ~27% de Churn (clase minoritaria). En vez de ajustar el umbral
    # DESPUÉS de entrenar, le decimos al modelo DESDE EL ENTRENAMIENTO que un error
    # en un cliente que sí se va (falso negativo) pesa más que un error en uno que se queda.
    # Peso = total_filas / (2 * filas_de_esa_clase) -> balancea ambas clases a peso "promedio" 1.0
    total = train_data.count()
    positivos = train_data.filter(col("label") == 1).count()
    negativos = total - positivos
    peso_positivo = total / (2.0 * positivos)
    peso_negativo = total / (2.0 * negativos)
    print(f"\nBalanceo de clases -> peso Churn=1: {peso_positivo:.3f} | peso Churn=0: {peso_negativo:.3f}")

    train_data = train_data.withColumn(
        "peso", when(col("label") == 1, peso_positivo).otherwise(peso_negativo)
    )

    # --- 2. Entrenamiento: Regresión Logística y Random Forest (con balanceo de clases) ---
    print("\nEntrenando Regresión Logística (con weightCol)...")
    lr = LogisticRegression(
        featuresCol="features", labelCol="label", weightCol="peso", maxIter=10, family="binomial"
    )
    lr_model = lr.fit(train_data)
    predictions_lr = lr_model.transform(test_data)

    print("Entrenando Random Forest (con weightCol)...")
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label", weightCol="peso", numTrees=100, maxDepth=8, seed=42
    )
    rf_model = rf.fit(train_data)
    predictions_rf = rf_model.transform(test_data)

    # --- 3. Evaluación ---
    resultado_lr = evaluar_modelo(predictions_lr, "Regresión Logística (weightCol, umbral 0.5)")
    resultado_rf = evaluar_modelo(predictions_rf, "Random Forest (weightCol, umbral 0.5)")

    # --- 4. Ajuste de umbral sobre la Regresión Logística ---
    predictions_lr_ajustado = predictions_lr.withColumn(
        "prob_churn", vector_to_array(col("probability"))[1]
    )
    predictions_lr_ajustado = predictions_lr_ajustado.withColumn(
        "prediction", (col("prob_churn") >= args.umbral).cast("double")
    )
    resultado_lr_ajustado = evaluar_modelo(
        predictions_lr_ajustado, f"Regresión Logística (weightCol, umbral {args.umbral})"
    )

    # --- 5. Seleccionar el mejor modelo por Recall ---
    candidatos = {
        "Regresión Logística (weightCol, umbral 0.5)": (resultado_lr, predictions_lr),
        "Random Forest (weightCol, umbral 0.5)": (resultado_rf, predictions_rf),
        f"Regresión Logística (weightCol, umbral {args.umbral})": (resultado_lr_ajustado, predictions_lr_ajustado),
    }
    mejor_nombre = max(candidatos, key=lambda k: candidatos[k][0]["recall"])
    mejor_resultado, mejor_predicciones = candidatos[mejor_nombre]
    print(f"\n[OK] Modelo seleccionado: {mejor_nombre} "
          f"(Recall={mejor_resultado['recall']:.4f}, Precision={mejor_resultado['precision']:.4f})")

    # --- 6. Armar el perfil de riesgo final por cliente (con datos de negocio) ---
    predicciones_finales = mejor_predicciones.withColumn(
        "probabilidad_churn", vector_to_array(col("probability"))[1]
    ).select(
        "customerID",
        col("label").alias("churn_real"),
        col("prediction").alias("churn_predicho"),
        "probabilidad_churn",
    ).join(df_negocio, on="customerID", how="left")

    df_perfil = predicciones_finales.toPandas()
    df_perfil["segmento_riesgo"] = df_perfil["probabilidad_churn"].apply(segmentar_riesgo)

    # --- 7. Exportar a CSV (para el dashboard / Power BI) ---
    os.makedirs(os.path.dirname(PREDICCIONES_OUTPUT), exist_ok=True)
    df_perfil.to_csv(PREDICCIONES_OUTPUT, index=False)
    print(f"[OK] Predicciones exportadas a: {PREDICCIONES_OUTPUT}")

    # --- 8. Importancia de variables (Random Forest) ---
    importancias = list(zip(COLUMNAS_FEATURES, rf_model.featureImportances.toArray()))
    importancias_df = pd.DataFrame(importancias, columns=["variable", "importancia"]) \
        .sort_values("importancia", ascending=False)
    importancias_df.to_csv(IMPORTANCIAS_OUTPUT, index=False)
    print(f"[OK] Importancia de variables exportada a: {IMPORTANCIAS_OUTPUT}")

    # --- 9. Guardar el perfil de riesgo en MongoDB (cierra el ciclo: Modelo -> NoSQL) ---
    registros = df_perfil.to_dict(orient="records")
    guardar_en_mongo(registros)

    spark.stop()


if __name__ == "__main__":
    main()
