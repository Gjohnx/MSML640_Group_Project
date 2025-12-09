# Training YOLOv11 on Google Vertex AI

## Generate the Dataset

```bash
python generate_images.py
```

## Configuration

- Project ID: msml640
- GS Bucket: msml640-dataset
- Region: us-central1

## Setup

### 1. Google Cloud SDK

```bash
brew install google-cloud-sdk
gcloud auth login
gcloud config set project msml640
```

### 2. Install Python Dependencies

```bash
pip install google-cloud-aiplatform google-cloud-storage
```

### 3. Build and Push Docker Image

Using a custom Docker image with pre-installed dependencies is faster and more reliable than installing dependencies at runtime.

#### 3.1. Create Artifact Registry Repository

```bash
export PROJECT_ID="msml640"
export REGION="us-central1"
export REPOSITORY="vertex-ai"

gcloud artifacts repositories create $REPOSITORY \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker images for Vertex AI training"
```

#### 3.2. Build and Push Docker Image

```bash
export PROJECT_ID="msml640"
export REGION="us-central1"
export REPOSITORY="vertex-ai"
export IMAGE_NAME="yolo11-rubiks-cube"
export IMAGE_TAG="latest"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
cd image/
docker build -t ${IMAGE_URI} -f Dockerfile .
docker push ${IMAGE_URI}
```

#### 3.3. Build using Google Cloud Build

```bash
cd image/
gcloud builds submit --tag ${IMAGE_URI} .
```

### 4. Upload the Dataset

```bash
export BUCKET_NAME="msml640-dataset"
export PROJECT_ID="msml640"
export REGION="us-central1"

gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME/
gsutil -m cp -r "dataset" gs://$BUCKET_NAME/datasets/
gsutil cp image/train_yolo_vertex.py gs://$BUCKET_NAME/scripts/ 
```

Update `data_gcs.yaml`:

```yaml
train: gs://msml640-dataset/datasets/dataset/train/images
val: gs://msml640-dataset/datasets/dataset/valid/images
test: gs://msml640-dataset/datasets/dataset/test/images
```

```bash
gsutil cp data_gcs.yaml gs://$BUCKET_NAME/datasets/dataset/data.yaml
```

### 5. Create Custom Job on Vertex AI

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=rubiks-cube-yolo11 \
  --config=config.yaml
```

The `config.yaml` uses the Docker image built in the previous step.

## Download the Trained Model

```bash
gsutil cp gs://$BUCKET_NAME/models/rubiks_cube_yolo11/weights/best.pt ./best.pt
```
