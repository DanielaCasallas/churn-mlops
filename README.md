# Churn Prediction Service

Servicio de machine learning que predice el **riesgo de deserción (churn)** de clientes de una empresa de telecomunicaciones, expuesto como API REST, empaquetado en Docker y verificado automáticamente en cada `push` vía GitHub Actions.

> **URL pública:** https://churn-mlops-27t8.onrender.com — [Swagger](https://churn-mlops-27t8.onrender.com/docs) · [/health](https://churn-mlops-27t8.onrender.com/health)

## Problema y datos

- **Problema**: clasificación binaria. Dado el perfil y contrato de un cliente, predecir la probabilidad de que abandone el servicio (`Churn`), con un umbral configurable.
- **Datos**: [Telco Customer Churn (IBM / Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), 7.043 clientes, 21 columnas. Datos públicos y sintéticos/anonimizados de ejemplo de IBM, sin información personal identificable.
- **Limpieza aplicada**: `TotalCharges` llega como texto y trae 11 filas con espacios en blanco (clientes con `tenure=0`, recién ingresados); se castean a numérico y esas 11 filas se descartan por no tener antigüedad ni facturación acumulada. `customerID` se excluye como feature por ser un identificador sin valor predictivo.

## Modelo

- **Pipeline**: `StandardScaler` (features numéricas) + `OneHotEncoder` (features categóricas) + `LogisticRegression` (`class_weight="balanced"` por el desbalance ~73/27 entre clientes que se quedan y que se van).
- **Entrenamiento reproducible**: seed fija (`random_state=42`), split 80/20 estratificado.
- **Métricas sobre el set de prueba (datos no vistos)**:

| Métrica | Valor |
|---|---|
| Accuracy | 0.7257 |
| Precision | 0.4901 |
| Recall | 0.7968 |
| F1 | 0.6069 |
| ROC-AUC | 0.8351 |

  Se priorizó **recall** alto (con `class_weight="balanced"`) porque en un caso de negocio real de retención de clientes, el costo de no detectar a alguien que se va (falso negativo) suele ser mayor que el de ofrecerle una retención de más a alguien que se iba a quedar igual (falso positivo).

## Cómo levantar el servicio

Requisitos: Docker y Docker Compose.

```bash
git clone <URL_DEL_REPOSITORIO>
cd <carpeta_del_repo>
docker compose up --build
```

El servicio queda escuchando en `http://localhost:8000`. Documentación interactiva (Swagger) en `http://localhost:8000/docs`.

> El modelo ya viene entrenado y versionado en `models/`. Si quieres reentrenarlo desde cero:
> ```bash
> pip install -r requirements-dev.txt
> python training/train.py
> python training/evaluate.py   # revalida las métricas sobre el mismo split
> ```

## Cómo probar

### `GET /health`

```bash
curl -s http://localhost:8000/health
```
```json
{"status":"ok","model_loaded":true,"model_type":"LogisticRegression"}
```

### `POST /predict`

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```
```json
{"churn_probability":0.8097,"churn_prediction":true,"threshold_used":0.5}
```

Se puede ajustar el umbral de decisión agregando `"threshold": 0.7` al body (por defecto es `0.5`).

### `POST /predict/batch`

Mismo contrato, envuelto en `{"instances": [ {...}, {...} ]}` (entre 1 y 500 instancias).

### `GET /model/schema`

```bash
curl -s http://localhost:8000/model/schema
```
```json
{"features":{"numeric":["tenure","MonthlyCharges","TotalCharges","SeniorCitizen"],"categorical":["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]},"target":"Churn","metrics":{"accuracy":0.7257,"precision":0.4901,"recall":0.7968,"f1":0.6069,"roc_auc":0.8351},"trained_at":"2026-08-16T18:40:58.474185+00:00"}
```

### Entrada inválida (422, no 500)

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"gender": "Female"}'
```
```
422
```

## Tests y linting

```bash
pip install -r requirements-dev.txt
pytest -v          # 13 tests, sin red ni credenciales
ruff check .        # sin hallazgos
```

## CI/CD

`.github/workflows/ci.yml` corre en cada `push`/`pull_request` a `main`, con jobs encadenados:

`lint` → `test` → `build` (imagen Docker) → `smoke` (levanta el contenedor real y consulta `/health` y `/predict` con `curl`) → `publish` (sube la imagen a GHCR, solo si se taggea `v*`).

## Despliegue (bonus)

El servicio está desplegado en **Render** (plan Free), a partir de la misma imagen Docker que se construye y valida en el pipeline de CI/CD.

- **URL pública**: https://churn-mlops-27t8.onrender.com
- **Proveedor y plan**: Render, Free tier.
- **Automatización**: el deploy lo dispara el job `deploy` de `.github/workflows/ci.yml`, que solo corre después de que pasen `lint` → `test` → `build` → `smoke`. Ese job llama al *Deploy Hook* de Render (guardado como `RENDER_DEPLOY_HOOK_URL` en GitHub Secrets, nunca en el código) y luego espera hasta que `/health` vuelva a responder en producción.
- **Limitación conocida**: el plan gratuito de Render "duerme" el servicio tras un período de inactividad. La primera request después de estar dormido puede demorar entre 20 y 50 segundos en responder mientras el contenedor arranca; las siguientes son inmediatas.

Prueba real contra producción:

```bash
curl -s https://churn-mlops-27t8.onrender.com/health
```
```json
{"status":"ok","model_loaded":true,"model_type":"LogisticRegression"}
```

## Variables de entorno

Ver `.env.example`. Ninguna variable requiere un valor secreto: el servicio no depende de credenciales externas para servir predicciones.

## Estructura del repositorio

```
app/            API FastAPI (rutas, contratos Pydantic, carga del modelo, config)
training/       Entrenamiento y evaluación, separado del servicio
models/         Artefactos versionados: model.joblib, preprocessor.joblib, metadata.json
tests/          pytest — API y pipeline de entrenamiento
data/           Dataset fuente (Telco Customer Churn)
docs/           Informe del proyecto
.github/workflows/ci.yml   Pipeline de CI/CD
```

## Limitaciones conocidas y qué haríamos con más tiempo

- El modelo es un `LogisticRegression` simple: prioriza velocidad de entrenamiento e interpretabilidad sobre performance máximo. Con más tiempo probaríamos `GradientBoosting`/`XGBoost` con búsqueda de hiperparámetros.
- La precision (0.49) es moderada por el `class_weight="balanced"`: el modelo prioriza no dejar pasar clientes en riesgo, a costa de más falsos positivos. Con más tiempo expondríamos el trade-off precision/recall como parámetro documentado en `/model/schema`.
- No hay reentrenamiento automático ni monitoreo de drift en producción — el modelo se sirve tal como quedó versionado en `models/` al momento del build.
- No se implementó autenticación en la API (fuera del alcance de la pauta, pero relevante para un despliegue real).

## Uso de asistentes de IA

Se usó **Claude (Anthropic)** como asistente durante el desarrollo del proyecto, en las siguientes partes:

- Generación inicial del código base: pipeline de entrenamiento (`training/train.py`, `evaluate.py`), API en FastAPI (`app/`), suite de tests (`tests/`), `Dockerfile`, `docker-compose.yml` y el workflow de CI/CD (`.github/workflows/ci.yml`).
- Redacción de la estructura inicial del `README.md` y de la plantilla del informe.
- Guía paso a paso para el flujo de Git/GitHub (creación del repo, primer commit, push) y para el diagnóstico de un conflicto de puerto al levantar el contenedor con Docker localmente.

Todo el código generado fue revisado y probado por el equipo (entrenamiento corrido localmente, tests ejecutados, `docker compose up` verificado en una máquina del equipo, pipeline de CI comprobado en verde en GitHub Actions) antes de la entrega. Cada integrante es responsable de poder explicar y defender en la exposición cualquier parte del código, independientemente de si fue escrita originalmente por una persona o generada con asistencia de IA.

_(Equipo: si además usaron otra herramienta — ChatGPT, GitHub Copilot, etc. — agréguenla aquí con el mismo nivel de detalle.)_

## Video de demostración

_(Completar: enlace al video de máximo 5 minutos mostrando el levantamiento del servicio, una predicción real y el pipeline en verde.)_

## Informe

Ver [`docs/informe.pdf`](docs/informe.pdf).
