# Informe — Churn Prediction Service

_Plantilla para el informe de 2-4 páginas exigido en `/docs/informe.pdf` (sección 04 de la pauta). Complétenla y expórtenla a PDF (Word, Google Docs, o pídanme que se las genere en Word directamente)._

## 1. Problema
- Qué se predice, para quién, por qué importa.
- Formato de entrada/salida (JSON de ejemplo).

## 2. Datos
- Fuente: Telco Customer Churn (IBM/Kaggle), 7043 clientes, 21 columnas.
- Limpieza aplicada (TotalCharges con 11 filas vacías descartadas, customerID excluido).
- Distribución del target (5174 No / 1869 Yes).

## 3. Modelo y decisiones de diseño
- Por qué LogisticRegression y no algo más complejo (velocidad, interpretabilidad, cumple con el foco de la pauta en ingeniería más que en sofisticación del modelo).
- Por qué class_weight="balanced" (desbalance de clases).
- Arquitectura del pipeline (preprocesador + modelo en un solo artefacto serializado).

## 4. Resultados
- Tabla de métricas sobre datos no vistos (accuracy, precision, recall, F1, ROC-AUC).
- Interpretación: ¿qué significan estos números para el negocio? (ej. de cada 10 clientes que realmente se van, el modelo detecta ~8 — recall 0.80 — a costa de falsos positivos).

## 5. Arquitectura del servicio
- Diagrama o descripción breve: training/ → models/ → app/ → Docker → CI/CD → (deploy).
- Decisiones de la API (validación Pydantic, manejo de errores, threshold configurable).

## 6. Limitaciones y trabajo futuro
- Ver sección correspondiente del README, ampliada con criterio propio del equipo.

## 7. Quién hizo qué
- Nombre — parte del proyecto (entrenamiento / API / Docker+CI / docs, etc.)

---
_Recuerden: cada integrante debe poder defender cualquier parte del código en la exposición, independiente de quién la haya escrito originalmente._
