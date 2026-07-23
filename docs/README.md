# Containerizing ML Models on ECS and Fargate

## Introduction

### What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting library designed for speed and performance. It builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. XGBoost is widely used for regression and classification tasks due to its ability to handle large datasets, manage missing values natively, and provide built-in regularization to prevent overfitting.

**How it works:** XGBoost uses a supervised learning approach called gradient boosted trees. The model consists of an ensemble of Classification and Regression Trees (CARTs), where the final prediction is the sum of the scores from all individual trees. Training follows an additive strategy — at each step, a new tree is added that optimizes a regularized objective function composed of a loss function (measuring prediction error) and a regularization term (penalizing model complexity via L1 and L2 norms). The algorithm uses gradient descent to minimize the loss, computing second-order Taylor approximation of the objective to efficiently find the best tree structure at each iteration. Key innovations include a sparsity-aware algorithm for handling missing values, a weighted quantile sketch for approximate split finding, and cache-aware block structures for out-of-core computation.

In this workshop, we use SageMaker's built-in XGBoost algorithm to train a regression model that predicts California housing prices based on features like median income, location, and housing characteristics. The trained model is exported as a Python pickle object, which can then be loaded into any Python environment for inference.

**References:**
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754) | [KDD 2016 PDF](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- [Introduction to Boosted Trees — XGBoost Official Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [How the SageMaker AI XGBoost Algorithm Works — AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-HowItWorks.html)
- [XGBoost Algorithm with Amazon SageMaker AI — AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)

### Deploying Other Models: ARIMA and SARIMA

The containerized hosting pattern demonstrated in this workshop is not limited to XGBoost. Any model that can be serialized (saved to disk) and loaded into a Python process can be deployed using the same approach. Two common examples from time-series forecasting are:

**ARIMA (AutoRegressive Integrated Moving Average):** A statistical model for analyzing and forecasting time-series data. ARIMA captures patterns in historical data by combining autoregression (past values), differencing (to make the series stationary), and moving average (past forecast errors). It's commonly used for demand forecasting, stock price prediction, and capacity planning.

**SARIMA (Seasonal ARIMA):** An extension of ARIMA that adds seasonal components to the model. SARIMA is ideal for data with repeating patterns at fixed intervals — such as monthly sales cycles, weekly traffic patterns, or annual temperature variations.

**How to adapt this workshop for ARIMA/SARIMA:**

1. **Train your model** using `statsmodels` (Python) and serialize it with `pickle` or `joblib`
2. **Replace the XGBoost model file** in the container artifacts with your ARIMA/SARIMA pickle file
3. **Modify the inference script** to load the model with `statsmodels` instead of `xgboost`, and adjust the prediction logic to accept time-series input
4. **Update `requirements.txt`** to include `statsmodels` instead of (or alongside) `xgboost`
5. **Build and deploy** using the same Docker, ECR, and ECS Fargate workflow

The core infrastructure pattern — Flask server in a container on ECS Fargate behind API Gateway — remains identical regardless of the ML framework or model type.

### AWS Services Used in This Workshop

| Service | Role in This Workshop |
|---------|----------------------|
| **Amazon SageMaker** | Fully managed ML service used here to run XGBoost training jobs on managed compute instances. It handles provisioning, training, and artifact storage automatically. |
| **Amazon S3** | Object storage for training data, validation data, and model artifacts (`model.tar.gz`). Acts as the durable interchange layer between training and deployment. |
| **Amazon ECR** | Private Docker container registry. Stores the inference container image so ECS can pull it during task launch. Supports vulnerability scanning on push. |
| **Amazon ECS** | Container orchestration service that manages task definitions, cluster configuration, and task lifecycle. Coordinates the deployment of your containerized model. |
| **AWS Fargate** | Serverless compute engine for ECS. Eliminates the need to provision or manage EC2 instances — you define CPU/memory requirements and Fargate handles the rest. |
| **Amazon API Gateway** | Managed API service that provides HTTPS endpoints, request throttling, and authentication. Front-ends the ECS task to expose the model as a production-ready API. |
| **Amazon CloudWatch** | Monitoring and logging service. Collects container logs via the `awslogs` driver and provides metrics for debugging and observability. |

---

## Workshop Overview

**Purpose:** This workshop teaches you how to take a trained ML model out of the notebook environment and into a production-ready, cost-effective hosting solution using containers. Rather than relying on always-on SageMaker inference endpoints, you will learn to package models into lightweight Docker containers and deploy them on serverless infrastructure — a pattern well-suited for models with sporadic traffic, limited size, and no GPU requirements.

**Target Audience:**
- Data scientists looking to move models from experimentation to production without managing infrastructure
- ML engineers seeking cost-effective alternatives to SageMaker real-time endpoints for simple models
- Cloud architects evaluating container-based inference patterns on AWS
- DevOps engineers who need to integrate ML models into existing ECS/Fargate workloads

**Expected knowledge:** Participants should be comfortable with Python, have basic familiarity with AWS services (S3, IAM), and understand fundamental ML concepts (training, inference, model serialization). No prior experience with Docker, ECS, or SageMaker is required — the workshop guides you through each step.

In this workshop, you will train an XGBoost regression model using Amazon SageMaker, package it into a Docker container, deploy it on Amazon ECS with AWS Fargate, and expose it as a REST API through Amazon API Gateway.

**Source Repository:** [UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway](https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)

**Based on:** [aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway) (updated for modern SageMaker runtimes and tooling)

**Duration:** ~2.5–3 hours (see time breakdown below)

**Level:** Intermediate (300)

**AWS Services Used:**
- Amazon SageMaker (Training)
- Amazon S3 (Data & Model Storage)
- Amazon ECR (Container Registry)
- Amazon ECS with AWS Fargate (Container Hosting)
- Amazon API Gateway (REST API)
- Amazon CloudWatch (Logging)

### Time Breakdown by Module

| Module | Hands-on Time | Wait/Provisioning Time | Total |
|--------|:-------------:|:----------------------:|:-----:|
| 1. Environment Setup | 10 min | 5 min (package installs) | ~15 min |
| 2. Data Preparation | 10 min | 2 min (S3 upload) | ~12 min |
| 3. Model Training | 5 min | 5–8 min (instance provisioning + training) | ~10–13 min |
| 4. Containerization | 10 min | 3–5 min (Docker build + ECR push) | ~13–15 min |
| 5. ECS Deployment | 10 min | 2–4 min (cluster + task provisioning) | ~12–14 min |
| 6. Testing | 10 min | 1–2 min (Security Group config) | ~11–12 min |
| 7. API Gateway | 15 min | 2 min (API deployment) | ~17 min |
| 8. Cleanup | 5 min | 1 min (resource deletion) | ~6 min |
| **Total** | **~75 min** | **~22–28 min** | **~100–110 min** |

> **Key wait times explained:**
> - **Training (~5–8 min):** SageMaker provisions the ml.m5.xlarge instance (~3 min), trains the model (~2 min on 20K records), and uploads artifacts to S3 (~1 min).
> - **Docker build (~3–5 min):** Pulling the Amazon Linux 2023 base image (~1 min), installing Python packages (~2 min), and pushing to ECR (~1–2 min depending on network).
> - **ECS Task startup (~2–4 min):** Fargate provisions the container runtime, pulls the image from ECR, starts the Flask server, and passes the health check (30-second intervals).

---

## Architecture

![Architecture Overview](diagrams/architecture-overview.drawio)

The solution follows this flow:

1. **Train** an XGBoost regression model on the California Housing dataset using SageMaker's built-in algorithm
2. **Package** the trained model into a Docker container with a Flask inference server
3. **Push** the container image to Amazon ECR
4. **Deploy** the container as an ECS Fargate task with a public IP
5. **Invoke** the model via HTTP POST requests (directly or through API Gateway)

> Open `diagrams/architecture-overview.drawio` and `diagrams/workflow-steps.drawio` in [draw.io](https://app.diagrams.net/) for detailed architecture and workflow diagrams.

---

## When to Use This Pattern

This pattern is ideal when you have:

- Simple models (< 200 MB)
- Sparse invocation patterns (no need for always-on inference instances)
- Models that do not require frequent retraining
- No GPU requirement for inference
- Need for a cost-effective, fully-managed, scalable solution

For models requiring auto-scaling, A/B testing, model monitoring, or GPU inference, consider using SageMaker real-time endpoints instead.

---

## Prerequisites

### AWS Account & Permissions

Your IAM execution role must have:

| Permission | Purpose |
|------------|---------|
| S3 Full Access (to your bucket) | Store training data and model artifacts |
| SageMaker Training | Launch training instances |
| CloudWatch Logs | Create log groups and write logs |
| ECR Full Access | Create repositories, push/pull images |
| ECS Full Access | Create clusters, register tasks, run tasks |
| EC2 Describe (Network Interfaces) | Retrieve public IPs of running tasks |
| IAM PassRole | Pass execution roles to ECS tasks |

### ECS Task Roles

You need two IAM roles for ECS:

1. **Task Role** (`ecsTaskRole`): The role the container itself assumes at runtime
2. **Task Execution Role** (`ecsTaskExecutionRole`): The role ECS uses to pull images and write logs
   - Minimum: `AmazonECSTaskExecutionRolePolicy` managed policy

### Environment

This workshop runs on either:
- **SageMaker Studio Space** (Ubuntu 24.04) - Recommended
- **SageMaker Notebook Instance** (Amazon Linux 2)

### Software Requirements

| Software | Minimum Version | Tested Version (Updated Baseline) |
|----------|----------------|----------------------------------|
| Python | 3.8+ | 3.12.13 |
| SageMaker SDK | sagemaker-core 2.x, sagemaker-train 1.x | sagemaker-core 2.13.1, sagemaker-train 1.13.1 |
| Boto3 | 1.28+ | 1.43.0 |
| AWS CLI | 2.x | 2.35.3 |
| Docker | 20.x+ | 29.5.3 |
| XGBoost (local) | 2.x | 2.1.4 |
| cURL | Any | 8.5.0 |

---

## Workshop Modules

### Module 1: Environment Setup

**Objective:** Prepare your SageMaker environment with all required tools.

**Steps:**

1. Open the Jupyter notebook: `notebooks/sm_xgboost_ca_housing_ecs_container_model_hosting.ipynb`

2. The notebook auto-detects your OS (Ubuntu or Amazon Linux) and adapts accordingly.

3. Verify installed software versions:
   ```python
   from importlib.metadata import version as pkg_version
   import boto3, sys, xgboost as xgb

   print(f"SageMaker SDK: sagemaker-core {pkg_version('sagemaker-core')}")
   print(f"Python: {sys.version}")
   print(f"Boto3: {boto3.__version__}")
   print(f"XGBoost: {xgb.__version__}")
   ```

4. Install and configure the Amazon ECR credential helper:
   - **Ubuntu:** `sudo apt-get install -y amazon-ecr-credential-helper`
   - **Amazon Linux:** `sudo yum install -y amazon-ecr-credential-helper`

5. Configure Docker to use the ECR credential helper:
   ```bash
   mkdir -p ~/.docker
   printf '{\n\t"credsStore": "ecr-login"\n}' > ~/.docker/config.json
   ```

6. Verify Docker is running:
   ```bash
   docker --version
   ```

> **Tip:** On SageMaker Spaces, you may need to start Docker manually with `sudo dockerd &`

---

### Module 2: Data Preparation

**Objective:** Load, split, standardize, and upload the California Housing dataset to S3.

**Dataset:** The [California Housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) contains 20,640 observations with 9 features:

| Feature | Description |
|---------|-------------|
| median_house_value | **Target variable** - Median house value |
| median_income | Median income in block group |
| housing_median_age | Median house age in block group |
| total_rooms | Total rooms in block group |
| total_bedrooms | Total bedrooms in block group |
| population | Population of block group |
| households | Number of households |
| latitude | Block group latitude |
| longitude | Block group longitude |

**Steps:**

1. Load the CSV file into a Pandas DataFrame:
   ```python
   pd_data_frame = pd.read_csv('datasets/california_housing.csv')
   ```

2. Split into train (80%), validation (8%), and test (12%) sets:
   ```python
   train, test = sklearn.model_selection.train_test_split(
       pd_data_frame, test_size=0.2, random_state=35, shuffle=True)
   train, val = sklearn.model_selection.train_test_split(
       train, test_size=0.1, random_state=25, shuffle=True)
   ```

3. Standardize features using `StandardScaler`:
   ```python
   scaler = StandardScaler()
   x_train = scaler.fit_transform(x_train)  # fit on train only
   x_val = scaler.transform(x_val)          # transform val/test
   x_test = scaler.transform(x_test)
   ```

4. Save locally and upload to S3:
   ```python
   train_dir_s3_path = sagemaker_session.upload_data(
       path='./data/.../train/', bucket=s3_bucket, key_prefix=train_dir_s3_prefix)
   ```

**S3 Structure:**
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

### Module 3: Model Training

**Objective:** Train an XGBoost regression model using SageMaker's built-in algorithm.

**Estimated time:** 10–13 minutes (5 min hands-on + 5–8 min waiting for training)

> **Timing breakdown:**
> - Instance provisioning: ~2–3 min (SageMaker launches and configures the ml.m5.xlarge)
> - Data download from S3 to instance: ~30 sec
> - XGBoost training (100 rounds on ~14,800 samples): ~1–2 min
> - Model upload to S3: ~30 sec
> - Instance teardown: ~1 min

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost 1.2-1 |
| Instance Type | ml.m5.xlarge |
| Instance Count | 1 |
| Objective | reg:squarederror |
| Max Depth | 6 |
| Learning Rate (eta) | 0.3 |
| Alpha (L1 reg) | 3 |
| Colsample by Tree | 0.7 |
| Num Rounds | 100 |

**Steps:**

1. Get the XGBoost training container image URI:
   ```python
   from sagemaker.core import image_uris

   container_image_uri = image_uris.retrieve(
       framework='xgboost', region=region_name,
       version='1.2-1', py_version='py37',
       instance_type='ml.m5.xlarge', image_scope='training')
   ```

2. Configure and launch training with ModelTrainer:
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

3. The trained model (`model.tar.gz`) is saved to S3 automatically.

> **Note:** The model is intentionally not tuned—this workshop focuses on the hosting pattern, not ML optimization.

---

### Module 4: Containerization

**Objective:** Package the trained model into a Docker container with a Flask inference server.

**Estimated time:** 13–15 minutes (10 min hands-on + 3–5 min build/push)

> **Timing breakdown:**
> - Model download from S3 & extraction: ~30 sec
> - Docker build (base image pull + pip install): ~2–4 min (first build; subsequent builds use cache)
> - ECR repository creation: ~5 sec
> - Docker push to ECR: ~1–2 min (image size ~400–500 MB)

**Steps:**

1. **Download and extract the model** from S3:
   ```python
   s3_bucket_resource.download_file(model_tar_file_s3_path_suffix, model_tar_file_local_path)
   with tarfile.open(model_tar_file_local_path, "r:gz") as tar:
       tar.extractall(path=container_artifacts_dir)
   ```

2. **Review the inference script** (`scripts/container_sm_xgboost_ca_housing_inference.py`):
   - Flask web server with two endpoints:
     - `POST /` — Accepts prediction requests
     - `GET /healthcheck` — Returns "OK" for ECS health checks
   - Loads model as a pickle object at startup
   - Accepts JSON input: `{"response_content_type": "...", "pred_x_csv": "val1,val2,..."}`

3. **Create the Dockerfile** (uses Amazon Linux 2023 as base):
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

4. **Build the Docker image:**
   ```bash
   docker build -t sm-xgboost-ca-housing-ecs-container-model-hosting container-artifacts/
   ```

5. **Create the ECR repository:**
   ```python
   ecr_client.create_repository(
       repositoryName=container_image_name,
       imageScanningConfiguration={'scanOnPush': True})
   ```

6. **Push to ECR:**
   ```bash
   docker tag <image>:latest <account>.dkr.ecr.<region>.amazonaws.com/<repo>:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/<repo>:latest
   ```

> **Security Note:** The ECR credential helper handles authentication automatically on Ubuntu/AL2/AL2023. No explicit `docker login` required.

---

### Module 5: ECS Fargate Deployment

**Objective:** Deploy the container as an ECS Fargate task.

**Estimated time:** 12–14 minutes (10 min hands-on + 2–4 min provisioning)

> **Timing breakdown:**
> - ECS cluster creation: ~10–30 sec (cluster goes ACTIVE almost immediately)
> - Task definition registration: ~5 sec
> - Task launch to RUNNING state: ~2–4 min
>   - Fargate container runtime provisioning: ~30 sec
>   - Image pull from ECR: ~30–60 sec
>   - Flask app startup + first health check pass: ~30–60 sec

**Steps:**

1. **Create the ECS cluster:**
   ```python
   ecs_client.create_cluster(clusterName=ecs_cluster_name)
   ```

2. **Register the Fargate task definition:**
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

3. **Run the task:**
   ```python
   ecs_client.run_task(
       cluster=ecs_cluster_name,
       count=1,
       launchType='FARGATE',
       networkConfiguration={
           'awsvpcConfiguration': {
               'subnets': ['<your-public-subnet>'],
               'securityGroups': ['<your-security-group>'],
               'assignPublicIp': 'ENABLED'
           }
       },
       taskDefinition=ecs_fargate_task_name
   )
   ```

**Key Configuration:**

| Parameter | Value |
|-----------|-------|
| Launch Type | FARGATE |
| CPU | 0.25 vCPU |
| Memory | 0.5 GB |
| Network Mode | awsvpc |
| Public IP | ENABLED |
| Container Port | 80 |
| Health Check | GET /healthcheck |

---

### Module 6: Testing the Deployment

**Objective:** Verify the model inference endpoint is working.

**Estimated time:** 11–12 minutes (10 min hands-on + 1–2 min for Security Group propagation)

> **Timing breakdown:**
> - Wait for task RUNNING (if not already): 0 sec (already waited in Module 5)
> - Security Group rule propagation: ~10–30 sec
> - Inference request latency (cold): ~200–500 ms (first request loads model into memory)
> - Inference request latency (warm): ~50–100 ms (subsequent requests)

**Steps:**

1. **Wait for task to reach RUNNING state** (~2-3 minutes):
   ```python
   while True:
       response = ecs_client.describe_tasks(cluster=ecs_cluster_name, tasks=[task_arn])
       status = response['tasks'][0]['lastStatus']
       if status in {'RUNNING', 'STOPPED'}:
           break
       time.sleep(5)
   ```

2. **Configure Security Group:** Add an inbound rule allowing HTTP (port 80) from your IP address.

3. **Get the task's public IP:**
   ```python
   # Retrieved from the task's Elastic Network Interface
   ecs_fargate_task_public_ip = '...'
   ```

4. **Send a prediction request:**
   ```bash
   curl -X POST -H 'Content-Type: application/json' \
     --data '{"response_content_type":"application/json","pred_x_csv":"0.12,-0.45,0.78,-0.23,0.56,-0.89,1.23,-0.67"}' \
     http://<TASK_PUBLIC_IP>:80/
   ```

5. **Expected response:**
   ```json
   {"Predicted value": "1.2345678"}
   ```

6. **Test the health check:**
   ```bash
   curl http://<TASK_PUBLIC_IP>:80/healthcheck
   # Response: OK
   ```

---

### Module 7: API Gateway Integration

**Objective:** Expose the inference endpoint as a managed HTTPS API.

For production-grade deployment, you will:

1. **Create an ECS Service** (instead of a standalone task) for multiple tasks with load balancing
2. **Setup an Application Load Balancer (ALB)** in front of the ECS Service
3. **Create an API Gateway HTTP or REST API** with the ALB as backend integration

**API Options:**

| Feature | HTTP API | REST API |
|---------|----------|----------|
| Cost | Lower | Higher |
| Latency | Lower | Higher |
| Features | Basic | Full (caching, request validation, WAF) |
| Use When | Simple proxy | Enterprise-grade API |

For guidance, see [Choosing between HTTP APIs and REST APIs](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-vs-rest.html).

---

### Module 8: Cleanup

**Objective:** Remove all resources to avoid ongoing charges.

**Steps (in order):**

```python
# 1. Stop the ECS Task
ecs_client.stop_task(cluster=ecs_cluster_name, task=ecs_fargate_task_id,
                     reason='Workshop cleanup')

# 2. Deregister the task definition
ecs_client.deregister_task_definition(taskDefinition=ecs_fargate_task_definiton_arn)

# 3. Delete the ECS cluster
ecs_client.delete_cluster(cluster=ecs_cluster_name)

# 4. Delete the ECR repository (force=True removes images)
ecr_client.delete_repository(repositoryName=container_image_name, force=True)

# 5. Delete S3 objects
for file in s3_bucket_resource.objects.filter(Prefix=f'{nb_name}/'):
    s3_resource.Object(s3_bucket_resource.name, file.key).delete()
```

**Additional cleanup:**
- Delete local Docker images: `docker system prune`
- If you created an API Gateway API, delete it from the console
- Delete CloudWatch Log Groups: `/ecs/<task-name>`

---

## Changelog: Updates from the Original aws-samples Repository

This project is an **updated baseline** of the [original aws-samples repository](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway), modernized to support current SageMaker runtimes, newer tooling, and broader environment compatibility.

The updated project is maintained at: [github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway](https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)

### Why the Update Was Needed

The original aws-samples project was written for SageMaker Notebook Instances running Amazon Linux with the legacy `sagemaker` Python SDK (`Estimator` API). Since then:

- **SageMaker Studio Spaces** (Ubuntu-based) became the recommended development environment
- The **SageMaker Python SDK** introduced a modular architecture (`sagemaker-core`, `sagemaker-train`)
- **Amazon Linux 2023** replaced Amazon Linux 2 as the default container base image
- The **ECR Docker Credential Helper** became the standard authentication method
- **Python 3.12** replaced Python 3.6/3.7 in SageMaker runtimes

### Summary of Changes

| Area | Original (aws-samples) | This Project (Updated Baseline) |
|------|------------------------|--------------------------------|
| Environment Support | SageMaker Notebook Instance only (Amazon Linux) | SageMaker Studio Spaces (Ubuntu 24.04) + Notebook Instances |
| OS Detection | `cat /etc/system-release` (Amazon Linux only) | `/etc/os-release` parsing (Ubuntu, AL2, AL2023) |
| SageMaker SDK | `import sagemaker` with `Estimator` API | Modular SDK: `sagemaker-core` 2.x, `sagemaker-train` 1.x |
| Training API | `Estimator(...).fit()` | `ModelTrainer(...).train()` with `InputData`, `Compute` configs |
| Python Runtime | 3.6.x | 3.12.x |
| XGBoost (local) | Not specified | 2.1.4 |
| Container Base Image | Amazon Linux 2 with `yum` | Amazon Linux 2023 with `dnf` + Python venv |
| Docker Auth | `docker login` with token (all OS) | ECR credential helper (auto) + fallback for legacy ALv1 |
| Package Installation | `yum` only | `apt` (Ubuntu) + `yum`/`dnf` (AL) with auto-detection |
| Docker in container | Flat pip install | Python virtual environment (`/opt/appenv`) |

### Detailed Changes

#### 1. Multi-OS Support (New in Updated Baseline)

**Original:** Only supported Amazon Linux v1/v2 on SageMaker notebook instances.

**Updated Baseline:** Automatically detects the host OS and adapts installation commands:
- Ubuntu 24.04 (SageMaker Spaces): Uses `apt-get`
- Amazon Linux 2 / AL2023 (Notebook Instances): Uses `yum` / `dnf`
- Fallback for unknown OS variants

#### 2. SageMaker SDK Migration (New in Updated Baseline)

**Original (legacy SDK):**
```python
import sagemaker
from sagemaker import Session, get_execution_role, image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

estimator = Estimator(image_uri=..., role=..., ...)
estimator.fit(inputs={'train': ..., 'validation': ...})
```

**Updated Baseline (modular SDK):**
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

#### 3. Container Image Update (New in Updated Baseline)

**Original Dockerfile used:**
```dockerfile
FROM public.ecr.aws/amazonlinux/amazonlinux:latest
RUN yum -y install python3 python3-pip
```

**Updated Baseline Dockerfile uses:**
```dockerfile
FROM public.ecr.aws/amazonlinux/amazonlinux:latest
RUN dnf -y install python3 python3-pip && dnf clean all
RUN python3 -m venv /opt/appenv
```

Key improvements:
- Uses `dnf` (Amazon Linux 2023 default package manager, replacing `yum`)
- Creates a Python virtual environment for cleaner dependency isolation
- Adds `dnf clean all` to reduce image size
- Uses `/opt/appenv/bin/python` as the entrypoint for proper venv activation

#### 4. ECR Authentication (New in Updated Baseline)

**Original:** Used explicit `docker login` with `aws ecr get-login-password` on all OS versions.

**Updated Baseline:** Uses the Amazon ECR Docker Credential Helper for seamless authentication:
- Installed via package manager (`apt` or `yum`)
- Configured in `~/.docker/config.json` with `"credsStore": "ecr-login"`
- Falls back to explicit login on legacy ALv1 systems

#### 5. Tested Runtime Versions (Updated Baseline)

The updated baseline has been tested and validated with:
- Python 3.12.13 (conda-forge)
- sagemaker-core 2.13.1
- sagemaker-train 1.13.1
- Boto3 1.43.0
- AWS CLI 2.35.3
- XGBoost 2.1.4 (local), 1.2-1 (training container)
- Docker 29.5.3
- Ubuntu 24.04 (SageMaker Studio Space)

#### 6. Unchanged Components

The following files remain **identical** to the original aws-samples repository:
- `notebooks/scripts/container_sm_xgboost_ca_housing_inference.py` (Flask inference server)
- `notebooks/scripts/container_sm_xgboost_ca_housing_inference_requirements.txt` (flask, pandas, xgboost)
- `notebooks/datasets/california_housing.csv` (Dataset)
- Overall solution architecture and workflow pattern
- XGBoost training algorithm version (1.2-1) and hyperparameters
- ECS Fargate deployment approach and task configuration

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Cannot connect to Docker daemon" | Run `sudo dockerd &` on SageMaker Spaces |
| ECR push fails with "no credentials" | Verify `~/.docker/config.json` has `"credsStore": "ecr-login"` |
| ECS Task goes to STOPPED | Check CloudWatch Logs at `/ecs/<task-name>` for container errors |
| Cannot reach task public IP | Verify Security Group allows inbound HTTP (port 80) from your IP |
| Training job fails | Verify IAM role has SageMaker training permissions and S3 access |
| "Module not found" errors | Restart the Jupyter kernel after installing packages |

### Useful Commands

```bash
# Check Docker daemon status
sudo systemctl status docker

# View ECS task logs
aws logs get-log-events --log-group-name /ecs/<task-name> --log-stream-name <stream>

# Describe running tasks
aws ecs describe-tasks --cluster <cluster-name> --tasks <task-arn>

# Test connectivity to task
curl -v http://<TASK_IP>:80/healthcheck
```

---

## Cost Estimation

| Resource | Duration/Usage | Approximate Cost |
|----------|---------------|-----------------|
| SageMaker Training (ml.m5.xlarge) | ~5–8 min | ~$0.03 |
| S3 Storage (training data + model) | < 10 MB | < $0.01 |
| ECR Storage (container image) | ~500 MB | ~$0.05/month |
| ECS Fargate Task (0.25 vCPU, 0.5 GB) | ~30 min (testing) | ~$0.01 |
| Data Transfer (ECR pull, S3) | < 1 GB | < $0.01 |
| CloudWatch Logs | < 1 MB | < $0.01 |
| API Gateway (if configured) | < 100 requests | < $0.01 |

**Total workshop cost:** < $0.50 (if cleaned up within one hour)

> **Note:** The primary cost driver is the SageMaker training instance. Since XGBoost trains quickly on this small dataset, actual training billable time is only ~5 minutes. ECS Fargate charges are per-second with a 1-minute minimum, so testing for 30 minutes costs only ~$0.01.

---

## Additional Resources

- [This Project (Updated Baseline)](https://github.com/UCdavidUC/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)
- [Original aws-samples Repository](https://github.com/aws-samples/amazon-sagemaker-xgboost-regression-model-hosting-on-amazon-ecs-fargate-and-amazon-api-gateway)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK (sagemaker-core)](https://sagemaker.readthedocs.io/en/stable/)
- [Amazon ECS on Fargate Documentation](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
- [Amazon API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)
- [XGBoost Algorithm Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)
- [Amazon ECR Docker Credential Helper](https://github.com/awslabs/amazon-ecr-credential-helper)

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file.
