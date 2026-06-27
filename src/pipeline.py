import os
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when
from pyspark.ml.feature import StringIndexer, VectorAssembler

CSV_PATH = "data/sample/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PARQUET_OUTPUT_PATH = "data/processed/telco_churn_procesado.parquet"

COLUMNAS_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_index", "Partner_index", "Dependents_index", "PhoneService_index",
    "MultipleLines_index", "InternetService_index", "OnlineSecurity_index",
    "OnlineBackup_index", "DeviceProtection_index", "TechSupport_index",
    "StreamingTV_index", "StreamingMovies_index", "Contract_index",
    "PaperlessBilling_index", "PaymentMethod_index"
]


def crear_sesion_spark() -> SparkSession:
    """Crea (o recupera) la sesión local de Spark para el pipeline."""
    spark = (
        SparkSession.builder
        .appName("TelcoChurn_DataPipeline")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    print(f"[OK] Sesión de Spark activa. Versión: {spark.version}")
    return spark


def ingesta(spark: SparkSession, csv_path: str = CSV_PATH):
    """Lee el dataset crudo desde CSV."""
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(csv_path)
    )
    print(f"[OK] Dataset cargado: {df.count()} filas, {len(df.columns)} columnas")
    return df


def limpiar(df):
    """
    Limpieza de TotalCharges.
    Decisión alineada con el EDA (01_exploracion_EDA.ipynb): los valores nulos
    corresponden a clientes con tenure=0 (recién ingresados, sin facturación
    todavía), por lo que se imputan con 0.0 y NO con la mediana.

    NOTA: en Spark 4.x (modo ANSI activado por defecto), convertir directamente
    un texto vacío (' ') a "double" con .cast("double") lanza un error
    (CAST_INVALID_INPUT) en vez de devolver NULL como en versiones anteriores.
    Por eso primero convertimos explícitamente los espacios en blanco a NULL,
    y recién después casteamos a double.
    """
    df_limpio = df.withColumn(
        "TotalCharges",
        when(trim(col("TotalCharges")) == "", None).otherwise(col("TotalCharges"))
    )
    df_limpio = df_limpio.withColumn("TotalCharges", col("TotalCharges").cast("double"))
    df_limpio = df_limpio.na.fill({"TotalCharges": 0.0})
    print("[OK] Limpieza de TotalCharges completada (imputación con 0.0)")
    return df_limpio


def indexar_categoricas(df):
    """Convierte todas las columnas de texto (excepto customerID) a índices numéricos."""
    columnas_categoricas = [
        item[0] for item in df.dtypes
        if item[1] == "string" and item[0] != "customerID"
    ]
    df_indexado = df
    for columna in columnas_categoricas:
        indexer = StringIndexer(
            inputCol=columna, outputCol=f"{columna}_index", handleInvalid="keep"
        )
        df_indexado = indexer.fit(df_indexado).transform(df_indexado)
    print(f"[OK] {len(columnas_categoricas)} columnas categóricas indexadas")
    return df_indexado


def ensamblar_features(df):
    """Junta todas las variables predictoras en un solo vector 'features'."""
    assembler = VectorAssembler(
        inputCols=COLUMNAS_FEATURES, outputCol="features", handleInvalid="skip"
    )
    df_final = assembler.transform(df)
    df_vectorizado = df_final.select(
        "customerID", "features", col("Churn_index").cast("int").alias("label")
    )
    print("[OK] Features ensamblados en vector único")
    return df_vectorizado


def guardar_parquet(df, ruta: str = PARQUET_OUTPUT_PATH):
    """Guarda el dataset procesado en formato Parquet (almacenamiento tipo Hadoop/HDFS local)."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.write.mode("overwrite").parquet(ruta)
    print(f"[OK] Datos procesados guardados en formato Parquet: {ruta}")


def correr_pipeline_completo(spark: SparkSession):
    """Corre ingesta -> limpieza -> indexación -> ensamblado -> guardado en Parquet."""
    df = ingesta(spark)
    df = limpiar(df)
    df = indexar_categoricas(df)
    df_vectorizado = ensamblar_features(df)
    guardar_parquet(df_vectorizado)
    return df_vectorizado


def main():
    parser = argparse.ArgumentParser(description="Pipeline de procesamiento Telco Churn")
    parser.add_argument(
        "--step",
        choices=["ingesta", "procesamiento", "todo"],
        default="todo",
        help="Qué etapa del pipeline ejecutar",
    )
    args = parser.parse_args()

    spark = crear_sesion_spark()

    if args.step == "ingesta":
        ingesta(spark)
    elif args.step == "procesamiento":
        df = ingesta(spark)
        df = limpiar(df)
        df = indexar_categoricas(df)
        df_vectorizado = ensamblar_features(df)
        guardar_parquet(df_vectorizado)
    else:  # "todo"
        correr_pipeline_completo(spark)

    spark.stop()


if __name__ == "__main__":
    main()
