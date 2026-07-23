# Containerizing ML Models on ECS and Fargate

## Introducción

### ¿Qué es XGBoost?

XGBoost (eXtreme Gradient Boosting) es una librería optimizada de gradient boosting diseñada para velocidad y rendimiento. Construye un conjunto de árboles de decisión de forma secuencial, donde cada nuevo árbol corrige los errores de los anteriores. XGBoost es ampliamente usado para tareas de regresión y clasificación por su capacidad de manejar datasets grandes, gestionar valores faltantes de forma nativa, y proveer regularización integrada para prevenir el sobreajuste.

**Cómo funciona:** XGBoost usa un enfoque de aprendizaje supervisado llamado gradient boosted trees. El modelo consiste en un ensamble de Árboles de Clasificación y Regresión (CARTs), donde la predicción final es la suma de los scores de todos los árboles individuales. El entrenamiento sigue una estrategia aditiva — en cada paso, se agrega un nuevo árbol que optimiza una función objetivo regularizada compuesta por una función de pérdida (que mide el error de predicción) y un término de regularización (que penaliza la complejidad del modelo con normas L1 y L2). El algoritmo usa gradient descent para minimizar la pérdida, calculando una aproximación de Taylor de segundo orden para encontrar de forma eficiente la mejor estructura de árbol en cada iteración. Sus innovaciones clave incluyen un algoritmo sparsity-aware para manejar valores faltantes, un weighted quantile sketch para encontrar splits aproximados, y estructuras de bloques cache-aware para procesamiento out-of-core.

En este workshop, usamos el algoritmo integrado de XGBoost de SageMaker para entrenar un modelo de regresión que predice precios de viviendas en California basándose en características como ingreso mediano, ubicación y características de las viviendas. El modelo entrenado se exporta como un objeto pickle de Python, que puede ser cargado en cualquier entorno Python para hacer inferencia.

**Referencias:**
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754) | [KDD 2016 PDF](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- [Introduction to Boosted Trees — Documentación Oficial de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [How the SageMaker AI XGBoost Algorithm Works — Documentación de AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html)
- [XGBoost Algorithm with Amazon SageMaker AI — Documentación de AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)

### Desplegando Otros Modelos: ARIMA y SARIMA

El patrón de deployment en contenedores que se demuestra en este workshop no se limita a XGBoost. Cualquier modelo que pueda ser serializado (guardado a disco) y cargado en un proceso de Python puede desplegarse con el mismo enfoque. Dos ejemplos comunes en pronóstico de series de tiempo son:

**ARIMA (AutoRegressive Integrated Moving Average):** Modelo estadístico para analizar y pronosticar series de tiempo. ARIMA captura patrones en datos históricos combinando autoregresión (valores pasados), diferenciación (para volver la serie estacionaria) y media móvil (errores anteriores del pronóstico). Se usa mucho para forecast de demanda, predicción de precios y planeación de capacidad.

**SARIMA (Seasonal ARIMA):** Extensión de ARIMA que agrega componentes estacionales. SARIMA es ideal para datos con patrones que se repiten a intervalos fijos — ciclos mensuales de ventas, patrones semanales de tráfico, o variaciones anuales de temperatura.

**Cómo adaptar este workshop para ARIMA/SARIMA:**

1. **Entrena tu modelo** con `statsmodels` (Python) y serialízalo con `pickle` o `joblib`
2. **Reemplaza el archivo del modelo XGBoost** en los artefactos del contenedor con tu pickle de ARIMA/SARIMA
3. **Modifica el script de inferencia** para cargar el modelo con `statsmodels` en vez de `xgboost`, y ajusta la lógica de predicción para recibir datos de series de tiempo
4. **Actualiza `requirements.txt`** para incluir `statsmodels` en lugar de (o junto con) `xgboost`
5. **Build y deploy** con el mismo flujo de Docker, ECR y ECS Fargate

El patrón de infraestructura — servidor Flask en un contenedor sobre ECS Fargate detrás de API Gateway — es el mismo sin importar qué framework de ML o tipo de modelo uses.

### Servicios de AWS Utilizados en Este Workshop

| Servicio | Rol en Este Workshop |
|----------|---------------------|
| **Amazon SageMaker** | Servicio de ML fully managed. Aquí se usa para correr jobs de entrenamiento de XGBoost en instancias de cómputo administradas. Se encarga del aprovisionamiento, entrenamiento y almacenamiento de artefactos de forma automática. |
| **Amazon S3** | Object storage para los datos de entrenamiento, validación y artefactos del modelo (`model.tar.gz`). Funciona como la capa de intercambio durable entre entrenamiento y deployment. |
| **Amazon ECR** | Registro privado de imágenes Docker. Almacena la imagen del contenedor de inferencia para que ECS la descargue al lanzar la tarea. Soporta vulnerability scanning al hacer push. |
| **Amazon ECS** | Servicio de orquestación de contenedores. Gestiona task definitions, configuración del cluster y ciclo de vida de las tareas. Coordina el deployment de tu modelo contenedorizado. |
| **AWS Fargate** | Motor de cómputo serverless para ECS. No necesitas aprovisionar ni administrar instancias EC2 — defines los requerimientos de CPU/memoria y Fargate se encarga del resto. |
| **Amazon API Gateway** | Servicio de APIs administrado que te da endpoints HTTPS, throttling de requests y autenticación. Se coloca frente a la tarea ECS para exponer el modelo como una API lista para producción. |
| **Amazon CloudWatch** | Servicio de monitoreo y logging. Recolecta los logs del contenedor con el driver `awslogs` y te da métricas para debugging y observabilidad. |

---

## Descripción General del Workshop

**Propósito:** Este workshop te enseña cómo llevar un modelo de ML entrenado desde el entorno de notebooks hasta una solución de alojamiento productiva y rentable usando contenedores. En lugar de depender de endpoints de inferencia de SageMaker que están activos todo el tiempo, aprenderás a empaquetar modelos en contenedores Docker ligeros y desplegarlos en infraestructura serverless — un patrón ideal para modelos con tráfico esporádico, tamaño limitado y sin necesidad de GPU.

**Audiencia Objetivo:**
- Data scientists que quieren llevar modelos de experimentación a producción sin administrar infraestructura
- Ingenieros de ML que buscan alternativas rentables a los endpoints real-time de SageMaker para modelos sencillos
- Arquitectos cloud que evalúan patrones de inferencia basados en contenedores en AWS
- Ingenieros DevOps que necesitan integrar modelos de ML en workloads existentes de ECS/Fargate

**Conocimiento esperado:** Los participantes deben manejar Python, tener familiaridad básica con servicios de AWS (S3, IAM), y entender conceptos fundamentales de ML (entrenamiento, inferencia, serialización de modelos). No se requiere experiencia previa con Docker, ECS o SageMaker — el workshop te lleva paso a paso.

En este workshop vas a entrenar un modelo de regresión XGBoost usando Amazon SageMaker, empaquetarlo en un contenedor Docker, desplegarlo en Amazon ECS con AWS Fargate, y exponerlo como una API REST a través de Amazon API Gateway.

**Repositorio Fuente:** [UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway](https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)

**Basado en:** [aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway) (actualizado para runtimes modernos de SageMaker y herramientas actuales)

**Duración:** ~2.5–3 horas (ver desglose de tiempos abajo)

**Nivel:** Intermedio (300)

**Servicios de AWS Utilizados:**
- Amazon SageMaker (Entrenamiento)
- Amazon S3 (Almacenamiento de Datos y Modelos)
- Amazon ECR (Registro de Contenedores)
- Amazon ECS con AWS Fargate (Alojamiento de Contenedores)
- Amazon API Gateway (API REST)
- Amazon CloudWatch (Logs)

### Desglose de Tiempos por Módulo

| Módulo | Tiempo Práctico | Tiempo de Espera/Aprovisionamiento | Total |
|--------|:---------------:|:----------------------------------:|:-----:|
| 1. Configuración del Entorno | 10 min | 5 min (instalación de paquetes) | ~15 min |
| 2. Preparación de Datos | 10 min | 2 min (carga a S3) | ~12 min |
| 3. Entrenamiento del Modelo | 5 min | 5–8 min (aprovisionamiento + entrenamiento) | ~10–13 min |
| 4. Contenedorización | 10 min | 3–5 min (build de Docker + push a ECR) | ~13–15 min |
| 5. Despliegue en ECS | 10 min | 2–4 min (aprovisionamiento del cluster + tarea) | ~12–14 min |
| 6. Pruebas | 10 min | 1–2 min (configuración de Security Group) | ~11–12 min |
| 7. API Gateway | 15 min | 2 min (despliegue del API) | ~17 min |
| 8. Limpieza | 5 min | 1 min (eliminación de recursos) | ~6 min |
| **Total** | **~75 min** | **~22–28 min** | **~100–110 min** |

> **Tiempos de espera explicados:**
> - **Entrenamiento (~5–8 min):** SageMaker aprovisiona la instancia ml.m5.xlarge (~3 min), entrena el modelo (~2 min con 20K registros), y sube los artefactos a S3 (~1 min).
> - **Build de Docker (~3–5 min):** Descarga de la imagen base Amazon Linux 2023 (~1 min), instalación de paquetes Python (~2 min), y push a ECR (~1–2 min según la red).
> - **Arranque de la tarea ECS (~2–4 min):** Fargate aprovisiona el runtime del contenedor, descarga la imagen de ECR, arranca el servidor Flask, y pasa el health check (intervalos de 30 segundos).

---

## Arquitectura

![Diagrama de Arquitectura](diagrams/architecture-overview.drawio)

La solución sigue este flujo:

1. **Entrenar** un modelo de regresión XGBoost con el dataset California Housing usando el algoritmo integrado de SageMaker
2. **Empaquetar** el modelo entrenado en un contenedor Docker con un servidor de inferencia Flask
3. **Subir** la imagen del contenedor a Amazon ECR
4. **Desplegar** el contenedor como una tarea ECS Fargate con IP pública
5. **Invocar** el modelo mediante peticiones HTTP POST (directamente o a través de API Gateway)

> Abre `diagrams/architecture-overview.drawio` y `diagrams/workflow-steps.drawio` en [draw.io](https://app.diagrams.net/) para ver los diagramas detallados de arquitectura y flujo de trabajo.

---

## Cuándo Usar Este Patrón

Este patrón es ideal cuando tienes:

- Modelos sencillos (< 200 MB)
- Patrones de invocación esporádicos (no necesitas instancias de inferencia activas todo el tiempo)
- Modelos que no requieren reentrenamiento frecuente
- Sin necesidad de GPU para inferencia
- Necesidad de una solución costo-efectiva, completamente administrada y escalable

Para modelos que requieren auto-scaling, pruebas A/B, monitoreo de modelos, o inferencia con GPU, considera usar endpoints de tiempo real de SageMaker.

---

## Prerrequisitos

### Cuenta de AWS y Permisos

Tu rol de ejecución IAM debe tener:

| Permiso | Propósito |
|---------|-----------|
| S3 Full Access (a tu bucket) | Almacenar datos de entrenamiento y artefactos del modelo |
| SageMaker Training | Lanzar instancias de entrenamiento |
| CloudWatch Logs | Crear grupos de logs y escribir logs |
| ECR Full Access | Crear repositorios, push/pull de imágenes |
| ECS Full Access | Crear clusters, registrar tareas, ejecutar tareas |
| EC2 Describe (Network Interfaces) | Obtener IPs públicas de las tareas |
| IAM PassRole | Pasar roles de ejecución a las tareas ECS |

### Roles para Tareas ECS

Necesitas dos roles IAM para ECS:

1. **Task Role** (`ecsTaskRole`): El rol que el contenedor asume en tiempo de ejecución
2. **Task Execution Role** (`ecsTaskExecutionRole`): El rol que ECS usa para descargar imágenes y escribir logs
   - Mínimo: política administrada `AmazonECSTaskExecutionRolePolicy`

### Entorno

Este workshop se ejecuta en cualquiera de estos entornos:
- **SageMaker Studio Space** (Ubuntu 24.04) - Recomendado
- **SageMaker Notebook Instance** (Amazon Linux 2)

### Requisitos de Software

| Software | Versión Mínima | Versión Probada (Baseline Actualizado) |
|----------|---------------|---------------------------------------|
| Python | 3.8+ | 3.12.13 |
| SageMaker SDK | sagemaker-core 2.x, sagemaker-train 1.x | sagemaker-core 2.13.1, sagemaker-train 1.13.1 |
| Boto3 | 1.28+ | 1.43.0 |
| AWS CLI | 2.x | 2.35.3 |
| Docker | 20.x+ | 29.5.3 |
| XGBoost (local) | 2.x | 2.1.4 |
| cURL | Cualquiera | 8.5.0 |

---

## Módulos del Workshop

### Módulo 1: Configuración del Entorno

**Objetivo:** Preparar tu entorno de SageMaker con todas las herramientas necesarias.

**Pasos:**

1. Abre el notebook de Jupyter: `notebooks/sm_xgboost_ca_housing_ecs_container_model_hosting.ipynb`

2. El notebook detecta automáticamente tu sistema operativo (Ubuntu o Amazon Linux) y se adapta.

3. Verifica las versiones del software instalado:
   ```python
   from importlib.metadata import version as pkg_version
   import boto3, sys, xgboost as xgb

   print(f"SageMaker SDK: sagemaker-core {pkg_version('sagemaker-core')}")
   print(f"Python: {sys.version}")
   print(f"Boto3: {boto3.__version__}")
   print(f"XGBoost: {xgb.__version__}")
   ```

4. Instala y configura el Amazon ECR credential helper:
   - **Ubuntu:** `sudo apt-get install -y amazon-ecr-credential-helper`
   - **Amazon Linux:** `sudo yum install -y amazon-ecr-credential-helper`

5. Configura Docker para usar el ECR credential helper:
   ```bash
   mkdir -p ~/.docker
   printf '{\n\t"credsStore": "ecr-login"\n}' > ~/.docker/config.json
   ```

6. Verifica que Docker esté corriendo:
   ```bash
   docker --version
   ```

> **Tip:** En SageMaker Spaces, puede que necesites arrancar Docker manualmente con `sudo dockerd &`

---

### Módulo 2: Preparación de Datos

**Objetivo:** Cargar, dividir, estandarizar y subir el dataset California Housing a S3.

**Dataset:** El [dataset California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) contiene 20,640 observaciones con 9 características:

| Característica | Descripción |
|----------------|-------------|
| median_house_value | **Variable objetivo** - Valor mediano de la vivienda |
| median_income | Ingreso mediano en el grupo de manzanas |
| housing_median_age | Edad mediana de las viviendas |
| total_rooms | Total de habitaciones |
| total_bedrooms | Total de recámaras |
| population | Población del grupo de manzanas |
| households | Número de hogares |
| latitude | Latitud del grupo de manzanas |
| longitude | Longitud del grupo de manzanas |

**Pasos:**

1. Carga el archivo CSV en un DataFrame de Pandas:
   ```python
   pd_data_frame = pd.read_csv('datasets/california_housing.csv')
   ```

2. Divide en conjuntos de entrenamiento (80%), validación (8%) y prueba (12%):
   ```python
   train, test = sklearn.model_selection.train_test_split(
       pd_data_frame, test_size=0.2, random_state=35, shuffle=True)
   train, val = sklearn.model_selection.train_test_split(
       train, test_size=0.1, random_state=25, shuffle=True)
   ```

3. Estandariza las características usando `StandardScaler`:
   ```python
   scaler = StandardScaler()
   x_train = scaler.fit_transform(x_train)  # fit solo en train
   x_val = scaler.transform(x_val)          # transform en val/test
   x_test = scaler.transform(x_test)
   ```

4. Guarda localmente y sube a S3:
   ```python
   train_dir_s3_path = sagemaker_session.upload_data(
       path='./data/.../train/', bucket=s3_bucket, key_prefix=train_dir_s3_prefix)
   ```

**Estructura en S3:**
```
s3://<bucket>/sm-xgboost-ca-housing-ecs-container-model-hosting/
├── data/
│   ├── train/train.csv
│   ├── validate/validate.csv
│   └── test/test.csv
├── checkpoint/
└── output/
```

---

### Módulo 3: Entrenamiento del Modelo

**Objetivo:** Entrenar un modelo de regresión XGBoost usando el algoritmo integrado de SageMaker.

**Tiempo estimado:** 10–13 minutos (5 min prácticos + 5–8 min esperando el entrenamiento)

> **Desglose de tiempos:**
> - Aprovisionamiento de la instancia: ~2–3 min (SageMaker lanza y configura la ml.m5.xlarge)
> - Descarga de datos de S3 a la instancia: ~30 seg
> - Entrenamiento XGBoost (100 rondas con ~14,800 muestras): ~1–2 min
> - Subida del modelo a S3: ~30 seg
> - Apagado de la instancia: ~1 min

**Configuración:**

| Parámetro | Valor |
|-----------|-------|
| Algoritmo | XGBoost 1.2-1 |
| Tipo de Instancia | ml.m5.xlarge |
| Número de Instancias | 1 |
| Objetivo | reg:squarederror |
| Profundidad Máxima | 6 |
| Tasa de Aprendizaje (eta) | 0.3 |
| Alpha (regularización L1) | 3 |
| Colsample by Tree | 0.7 |
| Número de Rondas | 100 |

**Pasos:**

1. Obtén el URI de la imagen del contenedor de entrenamiento XGBoost:
   ```python
   from sagemaker.core import image_uris

   container_image_uri = image_uris.retrieve(
       framework='xgboost', region=region_name,
       version='1.2-1', py_version='py37',
       instance_type='ml.m5.xlarge', image_scope='training')
   ```

2. Configura y lanza el entrenamiento con ModelTrainer:
   ```python
   from sagemaker.train import ModelTrainer
   from sagemaker.core.training.configs import Compute, InputData

   model_trainer = ModelTrainer(
       training_image=container_image_uri,
       hyperparameters=hyperparameters,
       compute=Compute(instance_type='ml.m5.xlarge', instance_count=1),
       output_data_config=OutputDataConfig(s3_output_path=model_output_s3_path),
       role=get_execution_role(),
       base_job_name=train_job_name,
       sagemaker_session=sagemaker_session
   )
   model_trainer.train(input_data_config=[train_data, val_data], wait=True)
   ```

3. El modelo entrenado (`model.tar.gz`) se guarda en S3 automáticamente.

> **Nota:** El modelo intencionalmente no está afinado — este workshop se enfoca en el patrón de deployment, no en la optimización del modelo de ML.

---

### Módulo 4: Contenedorización

**Objetivo:** Empaquetar el modelo entrenado en un contenedor Docker con un servidor de inferencia Flask.

**Tiempo estimado:** 13–15 minutos (10 min prácticos + 3–5 min de build/push)

> **Desglose de tiempos:**
> - Descarga del modelo de S3 y extracción: ~30 seg
> - Build de Docker (pull de imagen base + pip install): ~2–4 min (primera vez; builds posteriores usan caché)
> - Creación del repositorio ECR: ~5 seg
> - Push de Docker a ECR: ~1–2 min (imagen de ~400–500 MB)

**Pasos:**

1. **Descarga y extrae el modelo** de S3:
   ```python
   s3_bucket_resource.download_file(model_tar_file_s3_path_suffix, model_tar_file_local_path)
   with tarfile.open(model_tar_file_local_path, "r:gz") as tar:
       tar.extractall(path=container_artifacts_dir)
   ```

2. **Revisa el script de inferencia** (`scripts/container_sm_xgboost_ca_housing_inference.py`):
   - Servidor web Flask con dos endpoints:
     - `POST /` — Acepta peticiones de predicción
     - `GET /healthcheck` — Regresa "OK" para los health checks de ECS
   - Carga el modelo como objeto pickle al arrancar
   - Acepta entrada JSON: `{"response_content_type": "...", "pred_x_csv": "val1,val2,..."}`

3. **Crea el Dockerfile** (usa Amazon Linux 2023 como base):
   ```dockerfile
   FROM public.ecr.aws/amazonlinux/amazonlinux:latest
   WORKDIR /
   RUN dnf -y install python3 python3-pip && dnf clean all
   RUN python3 -m venv /opt/appenv
   COPY requirements.txt .
   RUN /opt/appenv/bin/pip install --no-cache-dir -r requirements.txt
   COPY xgboost-model ./
   COPY server.py ./
   ENV MODEL_PICKLE_FILE_PATH=xgboost-model
   ENV FLASK_SERVER_LOG_LEVEL=DEBUG
   ENV FLASK_SERVER_HOSTNAME=0.0.0.0
   ENV FLASK_SERVER_PORT=80
   ENV FLASK_SERVER_DEBUG=True
   ENTRYPOINT ["/opt/appenv/bin/python", "server.py"]
   ```

4. **Construye la imagen Docker:**
   ```bash
   docker build -t sm-xgboost-ca-housing-ecs-container-model-hosting container-artifacts/
   ```

5. **Crea el repositorio en ECR:**
   ```python
   ecr_client.create_repository(
       repositoryName=container_image_name,
       imageScanningConfiguration={'scanOnPush': True})
   ```

6. **Sube a ECR:**
   ```bash
   docker tag <imagen>:latest <cuenta>.dkr.ecr.<region>.amazonaws.com/<repo>:latest
   docker push <cuenta>.dkr.ecr.<region>.amazonaws.com/<repo>:latest
   ```

> **Nota de Seguridad:** El ECR credential helper maneja la autenticación automáticamente en Ubuntu/AL2/AL2023. No se necesita un `docker login` explícito.

---

### Módulo 5: Despliegue en ECS Fargate

**Objetivo:** Desplegar el contenedor como una tarea de ECS Fargate.

**Tiempo estimado:** 12–14 minutos (10 min prácticos + 2–4 min de aprovisionamiento)

> **Desglose de tiempos:**
> - Creación del cluster ECS: ~10–30 seg (el cluster pasa a ACTIVE casi de inmediato)
> - Registro de la definición de tarea: ~5 seg
> - Lanzamiento de tarea hasta estado RUNNING: ~2–4 min
>   - Aprovisionamiento del runtime del contenedor Fargate: ~30 seg
>   - Pull de la imagen desde ECR: ~30–60 seg
>   - Arranque de la app Flask + primer health check exitoso: ~30–60 seg

**Pasos:**

1. **Crea el cluster de ECS:**
   ```python
   ecs_client.create_cluster(clusterName=ecs_cluster_name)
   ```

2. **Registra la definición de tarea Fargate:**
   ```python
   ecs_client.register_task_definition(
       family=ecs_fargate_task_name,
       taskRoleArn=ecs_fargate_task_role,
       executionRoleArn=ecs_fargate_task_execution_role,
       networkMode='awsvpc',
       containerDefinitions=[{
           'name': ecs_container_name,
           'image': target_image_name,
           'portMappings': [{'containerPort': 80, 'hostPort': 80, 'protocol': 'tcp'}],
           'logConfiguration': {
               'logDriver': 'awslogs',
               'options': {
                   'awslogs-create-group': 'true',
                   'awslogs-region': region_name,
                   'awslogs-group': f'/ecs/{ecs_fargate_task_name}',
                   'awslogs-stream-prefix': 'ecs'
               }
           },
           'healthCheck': {
               'command': ["CMD-SHELL", "curl -f http://localhost:80/healthcheck || exit 1"],
               'interval': 30,
               'timeout': 30
           }
       }],
       requiresCompatibilities=['FARGATE'],
       cpu='0.25 vCPU',
       memory='0.5 GB'
   )
   ```

3. **Ejecuta la tarea:**
   ```python
   ecs_client.run_task(
       cluster=ecs_cluster_name,
       count=1,
       launchType='FARGATE',
       networkConfiguration={
           'awsvpcConfiguration': {
               'subnets': ['<tu-subnet-publica>'],
               'securityGroups': ['<tu-security-group>'],
               'assignPublicIp': 'ENABLED'
           }
       },
       taskDefinition=ecs_fargate_task_name
   )
   ```

**Configuración Clave:**

| Parámetro | Valor |
|-----------|-------|
| Tipo de Lanzamiento | FARGATE |
| CPU | 0.25 vCPU |
| Memoria | 0.5 GB |
| Modo de Red | awsvpc |
| IP Pública | ENABLED |
| Puerto del Contenedor | 80 |
| Health Check | GET /healthcheck |

---

### Módulo 6: Pruebas del Despliegue

**Objetivo:** Verificar que el endpoint de inferencia del modelo funciona correctamente.

**Tiempo estimado:** 11–12 minutos (10 min prácticos + 1–2 min de propagación del Security Group)

> **Desglose de tiempos:**
> - Espera a estado RUNNING (si no se completó antes): 0 seg (ya se esperó en el Módulo 5)
> - Propagación de la regla del Security Group: ~10–30 seg
> - Latencia de la petición de inferencia (en frío): ~200–500 ms (primera petición carga el modelo en memoria)
> - Latencia de la petición de inferencia (en caliente): ~50–100 ms (peticiones subsecuentes)

**Pasos:**

1. **Espera a que la tarea llegue al estado RUNNING** (~2-3 minutos):
   ```python
   while True:
       response = ecs_client.describe_tasks(cluster=ecs_cluster_name, tasks=[task_arn])
       status = response['tasks'][0]['lastStatus']
       if status in {'RUNNING', 'STOPPED'}:
           break
       time.sleep(5)
   ```

2. **Configura el Security Group:** Agrega una regla de entrada que permita HTTP (puerto 80) desde tu dirección IP.

3. **Obtén la IP pública de la tarea:**
   ```python
   # Se obtiene de la Elastic Network Interface de la tarea
   ecs_fargate_task_public_ip = '...'
   ```

4. **Envía una petición de predicción:**
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
     --data '{"response_content_type":"application/json","pred_x_csv":"0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67"}' \
     http://<IP_PUBLICA_TAREA>:80/
   ```

5. **Respuesta esperada:**
   ```json
   {"Predicted value": "1.2345678"}
   ```

6. **Prueba el health check:**
   ```bash
   curl http://<IP_PUBLICA_TAREA>:80/healthcheck
   # Respuesta: OK
   ```

---

### Módulo 7: Integración con API Gateway

**Objetivo:** Exponer el endpoint de inferencia como una API HTTPS administrada.

Para un despliegue de grado producción, vas a:

1. **Crear un ECS Service** (en lugar de una tarea independiente) para tener múltiples tareas con balanceo de carga
2. **Configurar un Application Load Balancer (ALB)** frente al ECS Service
3. **Crear un API Gateway HTTP o REST API** con el ALB como integración backend

**Opciones de API:**

| Característica | HTTP API | REST API |
|----------------|----------|----------|
| Costo | Menor | Mayor |
| Latencia | Menor | Mayor |
| Funcionalidades | Básicas | Completas (caché, validación de requests, WAF) |
| Usar cuando | Proxy simple | API de grado empresarial |

Para más detalles, consulta [Elegir entre HTTP APIs y REST APIs](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-vs-rest.html).

---

### Módulo 8: Limpieza

**Objetivo:** Eliminar todos los recursos para evitar cargos continuos.

**Pasos (en orden):**

```python
# 1. Detener la tarea ECS
ecs_client.stop_task(cluster=ecs_cluster_name, task=ecs_fargate_task_id,
                     reason='Limpieza del workshop')

# 2. Desregistrar la definición de tarea
ecs_client.deregister_task_definition(taskDefinition=ecs_fargate_task_definiton_arn)

# 3. Eliminar el cluster ECS
ecs_client.delete_cluster(cluster=ecs_cluster_name)

# 4. Eliminar el repositorio ECR (force=True elimina las imágenes)
ecr_client.delete_repository(repositoryName=container_image_name, force=True)

# 5. Eliminar objetos de S3
for file in s3_bucket_resource.objects.filter(Prefix=f'{nb_name}/'):
    s3_resource.Object(s3_bucket_resource.name, file.key).delete()
```

**Limpieza adicional:**
- Elimina imágenes locales de Docker: `docker system prune`
- Si creaste un API Gateway, elimínalo desde la consola
- Elimina los Log Groups de CloudWatch: `/ecs/<nombre-tarea>`

---

## Registro de Cambios: Actualizaciones del Repositorio Original aws-samples

Este proyecto es una **versión base actualizada** del [repositorio original aws-samples](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway), modernizado para soportar los runtimes actuales de SageMaker, herramientas más nuevas, y mayor compatibilidad de entornos.

El proyecto actualizado se mantiene en: [github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway](https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)

### Por Qué Se Necesitó la Actualización

El proyecto original de aws-samples fue escrito para SageMaker Notebook Instances corriendo Amazon Linux con el SDK legacy de `sagemaker` (API de `Estimator`). Desde entonces:

- **SageMaker Studio Spaces** (basado en Ubuntu) se convirtió en el entorno de desarrollo recomendado
- El **SDK de SageMaker para Python** introdujo una arquitectura modular (`sagemaker-core`, `sagemaker-train`)
- **Amazon Linux 2023** reemplazó a Amazon Linux 2 como imagen base predeterminada de contenedores
- El **ECR Docker Credential Helper** se volvió el método estándar de autenticación
- **Python 3.12** reemplazó a Python 3.6/3.7 en los runtimes de SageMaker

### Resumen de Cambios

| Área | Original (aws-samples) | Este Proyecto (Baseline Actualizado) |
|------|------------------------|-------------------------------------|
| Soporte de Entorno | Solo SageMaker Notebook Instance (Amazon Linux) | SageMaker Studio Spaces (Ubuntu 24.04) + Notebook Instances |
| Detección de SO | `cat /etc/system-release` (solo Amazon Linux) | Parseo de `/etc/os-release` (Ubuntu, AL2, AL2023) |
| SDK de SageMaker | `import sagemaker` con API de `Estimator` | SDK modular: `sagemaker-core` 2.x, `sagemaker-train` 1.x |
| API de Entrenamiento | `Estimator(...).fit()` | `ModelTrainer(...).train()` con configs `InputData`, `Compute` |
| Runtime de Python | 3.6.x | 3.12.x |
| XGBoost (local) | No especificado | 2.1.4 |
| Imagen Base del Contenedor | Amazon Linux 2 con `yum` | Amazon Linux 2023 con `dnf` + venv de Python |
| Autenticación Docker | `docker login` con token (todos los SO) | ECR credential helper (auto) + fallback para ALv1 legacy |
| Instalación de Paquetes | Solo `yum` | `apt` (Ubuntu) + `yum`/`dnf` (AL) con auto-detección |
| Docker en contenedor | pip install directo | Entorno virtual de Python (`/opt/appenv`) |

### Cambios Detallados

#### 1. Soporte Multi-SO (Nuevo en el Baseline Actualizado)

**Original:** Solo soportaba Amazon Linux v1/v2 en SageMaker notebook instances.

**Baseline Actualizado:** Detecta automáticamente el SO del host y adapta los comandos de instalación:
- Ubuntu 24.04 (SageMaker Spaces): Usa `apt-get`
- Amazon Linux 2 / AL2023 (Notebook Instances): Usa `yum` / `dnf`
- Fallback para variantes de SO desconocidas

#### 2. Migración del SDK de SageMaker (Nuevo en el Baseline Actualizado)

**Original (SDK legacy):**
```python
import sagemaker
from sagemaker import Session, get_execution_role, image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

estimator = Estimator(image_uri=..., role=..., ...)
estimator.fit(inputs={'train': ..., 'validation': ...})
```

**Baseline Actualizado (SDK modular):**
```python
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import image_uris
from sagemaker.core.inputs import TrainingInput
from sagemaker.train import ModelTrainer
from sagemaker.core.training.configs import Compute, InputData

model_trainer = ModelTrainer(
    training_image=container_image_uri,
    hyperparameters=hyperparameters,
    compute=Compute(instance_type=..., instance_count=...),
    ...
)
model_trainer.train(input_data_config=[train_data, val_data], wait=True)
```

#### 3. Actualización de la Imagen del Contenedor (Nuevo en el Baseline Actualizado)

**Dockerfile original usaba:**
```dockerfile
FROM public.ecr.aws/amazonlinux/amazonlinux:latest
RUN yum -y install python3 python3-pip
```

**Dockerfile del Baseline Actualizado usa:**
```dockerfile
FROM public.ecr.aws/amazonlinux/amazonlinux:latest
RUN dnf -y install python3 python3-pip && dnf clean all
RUN python3 -m venv /opt/appenv
```

Mejoras clave:
- Usa `dnf` (gestor de paquetes predeterminado de Amazon Linux 2023, reemplazando a `yum`)
- Crea un entorno virtual de Python para mejor aislamiento de dependencias
- Agrega `dnf clean all` para reducir el tamaño de la imagen
- Usa `/opt/appenv/bin/python` como entrypoint para activación correcta del venv

#### 4. Autenticación con ECR (Nuevo en el Baseline Actualizado)

**Original:** Usaba `docker login` explícito con `aws ecr get-login-password` en todas las versiones de SO.

**Baseline Actualizado:** Usa el Amazon ECR Docker Credential Helper para autenticación transparente:
- Se instala via gestor de paquetes (`apt` o `yum`)
- Se configura en `~/.docker/config.json` con `"credsStore": "ecr-login"`
- Hace fallback a login explícito en sistemas legacy ALv1

#### 5. Versiones de Runtime Probadas (Baseline Actualizado)

El baseline actualizado ha sido probado y validado con:
- Python 3.12.13 (conda-forge)
- sagemaker-core 2.13.1
- sagemaker-train 1.13.1
- Boto3 1.43.0
- AWS CLI 2.35.3
- XGBoost 2.1.4 (local), 1.2-1 (contenedor de entrenamiento)
- Docker 29.5.3
- Ubuntu 24.04 (SageMaker Studio Space)

#### 6. Componentes Sin Cambios

Los siguientes archivos permanecen **idénticos** al repositorio original de aws-samples:
- `notebooks/scripts/container_sm_xgboost_ca_housing_inference.py` (Servidor de inferencia Flask)
- `notebooks/scripts/container_sm_xgboost_ca_housing_inference_requirements.txt` (flask, pandas, xgboost)
- `notebooks/datasets/california_housing.csv` (Dataset)
- Arquitectura y patrón de flujo de trabajo general de la solución
- Versión del algoritmo de entrenamiento XGBoost (1.2-1) e hiperparámetros
- Enfoque de despliegue en ECS Fargate y configuración de tareas

---

## Solución de Problemas

### Problemas Comunes

| Problema | Solución |
|----------|----------|
| "Cannot connect to Docker daemon" | Ejecuta `sudo dockerd &` en SageMaker Spaces |
| Push a ECR falla con "no credentials" | Verifica que `~/.docker/config.json` tenga `"credsStore": "ecr-login"` |
| La tarea ECS pasa a STOPPED | Revisa los logs de CloudWatch en `/ecs/<nombre-tarea>` |
| No se puede alcanzar la IP pública de la tarea | Verifica que el Security Group permita HTTP (puerto 80) desde tu IP |
| El training job falla | Verifica que el rol IAM tenga permisos de entrenamiento en SageMaker y acceso a S3 |
| Errores de "Module not found" | Reinicia el kernel de Jupyter después de instalar paquetes |

### Comandos Útiles

```bash
# Verificar el estado del daemon de Docker
sudo systemctl status docker

# Ver logs de la tarea ECS
aws logs get-log-events --log-group-name /ecs/<nombre-tarea> --log-stream-name <stream>

# Describir tareas en ejecución
aws ecs describe-tasks --cluster <nombre-cluster> --tasks <arn-tarea>

# Probar conectividad a la tarea
curl -v http://<IP_TAREA>:80/healthcheck
```

---

## Estimación de Costos

| Recurso | Duración/Uso | Costo Aproximado |
|---------|-------------|-----------------|
| SageMaker Training (ml.m5.xlarge) | ~5–8 min | ~$0.03 USD |
| Almacenamiento S3 (datos + modelo) | < 10 MB | < $0.01 USD |
| Almacenamiento ECR (imagen de contenedor) | ~500 MB | ~$0.05 USD/mes |
| Tarea ECS Fargate (0.25 vCPU, 0.5 GB) | ~30 min (pruebas) | ~$0.01 USD |
| Transferencia de Datos (pull de ECR, S3) | < 1 GB | < $0.01 USD |
| CloudWatch Logs | < 1 MB | < $0.01 USD |
| API Gateway (si se configura) | < 100 peticiones | < $0.01 USD |

**Costo total del workshop:** < $0.50 USD (si limpias los recursos dentro de una hora)

> **Nota:** El principal generador de costo es la instancia de entrenamiento de SageMaker. Como XGBoost entrena rápido con este dataset pequeño, el tiempo facturable real es solo ~5 minutos. Los cargos de ECS Fargate son por segundo con un mínimo de 1 minuto, así que probar por 30 minutos cuesta solo ~$0.01 USD.

---

## Recursos Adicionales

- [Este Proyecto (Baseline Actualizado)](https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)
- [Repositorio Original aws-samples](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)
- [Documentación de Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/)
- [SDK de SageMaker para Python (sagemaker-core)](https://sagemaker.readthedocs.io/en/stable/)
- [Documentación de Amazon ECS en Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [Documentación de Amazon API Gateway](https://docs.aws.amazon.com/apigateway/)
- [Documentación del Algoritmo XGBoost](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [Amazon ECR Docker Credential Helper](https://github.com/awslabs/amazon-ecr-credential-helper)

---

## Licencia

Esta librería está licenciada bajo la Licencia MIT-0. Consulta el archivo [LICENSE](../LICENSE).
