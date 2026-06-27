# 📱Predicción de Churn en Telecomunicaciones 📱 
# Big Data DD283

Universidad Autónoma del Perú | Semestre 2026-1

👥 Equipo

| Nombre | GitHub | Rol en el proyecto |
| :--- | :--- | :--- |
| **[Alexis velarde Bruno]** | @ALEXISBRUNO-code | Líder + Ingeniería de Datos |
| [Jesus Tolentino Vargas] | @J3susTo| Machine Learning |
| [Giancarlos Alegria Ibarra] | @giancarlosalegria39 | Visualización + Presentación |

🎯 Descripción del Proyecto

El Problema
El mercado de telecomunicaciones en el Perú (Claro, Movistar, Entel) es altamente competitivo. Adquirir un cliente nuevo en la modalidad postpago cuesta entre 5 y 7 veces más que retener a uno existente. Actualmente, las operadoras reportan pérdidas millonarias debido a la fuga masiva de clientes (Churn) hacia la competencia mediante la portabilidad numérica. Las empresas suelen reaccionar de manera tardía cuando el usuario ya inició el trámite de baja ante OSIPTEL, perdiendo la oportunidad de retenerlo.

El Objetivo
Construir un pipeline de Big Data utilizando Python y PySpark capaz de predecir con 30 días de anticipación qué clientes de telefonía postpago/control tienen una alta probabilidad de cancelar su servicio o portarse a otra compañía, logrando un Recall > 80% en las detecciones para ejecutar campañas de retención proactivas.

¿Por qué Big Data?
* **Volumen:** Estimación de un histórico simulado de 500,000 clientes activos que generan millones de registros mensuales de llamadas, consumo de gigas y logs de red.
* **Velocidad:** Procesamiento analítico en Batch (Lotes) ejecutado de forma diaria.
* **Variedad:** Integración de datos estructurados del CRM (planes, facturación, pagos) y semiestructurados (historial de quejas en texto y logs de navegación).
* **Veracidad:** Manejo y limpieza de valores nulos por llamadas caídas, reclamos duplicados y desfases en los ciclos de facturación de los clientes.
* **Valor:** Identificación temprana de clientes en riesgo para reducir el Churn mensual en un 5%, protegiendo el ARPU (ingreso promedio por usuario) de la empresa.

🏗️ Arquitectura

[Fuentes de Datos / CRM] → [Archivos Parquet/CSV] → [Apache Spark (ETL)] → [PySpark MLlib (Modelamiento)] → [MongoDB Atlas] → [Power BI / Dashboard]

🛠️ Stack Tecnológico

| Capa | Tecnología | Propósito |
| :--- | :--- | :--- |
| Ingesta | Python (Faker) | Generar un dataset masivo de clientes sintéticos con variables del mercado peruano. |
| Almacenamiento | Hadoop HDFS + Parquet | Almacenar localmente los datos crudos y procesados de manera distribuida y eficiente. |
| Procesamiento | Apache Spark 3.5 | Realizar la limpieza, uniones de tablas y agregaciones de datos a gran escala. |
| ML | PySpark MLlib | Entrenar y evaluar modelos de clasificación binaria (Random Forest o GBT Classifier). |
| Visualización | Power BI | Construir un dashboard interactivo para el equipo de marketing con las alertas de fuga. |

📁 Estructura del Repositorio
```bash
proyecto-telco-churn/
├── notebooks/
│   ├── 01_exploracion_EDA.ipynb
│   ├── 02_pipeline_procesamiento.ipynb
│   ├── 03_modelo_ml.ipynb
│   └── 04_dashboard.ipynb
├── src/
│   ├── pipeline.py
│   └── modelo.py
├── data/
│   └── sample/          
├── docs/
│   ├── arquitectura.drawio
│   └── presentacion_ep.pdf
└── requirements.txt
```

🚀 Cómo Ejecutar

Pre-requisitos

```bash
pip install -r requirements.txt
```
Configuración
```bash
# Crear archivo .env con tus credenciales:
cp .env.example .env
# Editar .env con tus keys de MongoDB, etc.
```
Ejecutar el pipeline
```bash
# Paso 1: Ingesta de datos
python src/pipeline.py --step ingesta

# Paso 2: Procesamiento
jupyter notebook notebooks/02_pipeline_procesamiento.ipynb

# Paso 3: Modelo ML
jupyter notebook notebooks/03_modelo_ml.ipynb
```

📅 Progreso Semanal

| Semana | Entregable                     | Estado      |
|--------|--------------------------------|-------------|
| S1     | Arquitectura + EDA inicial     | REALIZADO |
| S2     | Pipeline Hadoop/Spark          | REALIZADO |
| S3     | Implementación NoSQL           | REALIAZDO |
| S4     | Sustentación EP                | REALIZADO |
| S5     | Procesamiento Spark completo   | ⬜ Pendiente |
| S6     | Modelo ML entrenado            | ⬜ Pendiente |
| S7     | Datos limpios + Scraping       | ⬜ Pendiente |
| S8     | Sustentación EF                | ⬜ Pendiente |

# 📚 Referencias y Datasets

IBM Telco Customer Churn Dataset — https://github.com/IBM/telco-customer-churn-on-icp4d

Documentación de PySpark MLlib — https://spark.apache.org/docs/latest/ml-guide.html

OSIPTEL Perú Portal de Datos — https://www.osiptel.gob.pe/

Big Data DD283 | Universidad Autónoma del Perú | 2026-1