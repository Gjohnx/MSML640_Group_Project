# Training CNN on Google Vertex AI

## Configuration

- Project ID: msml640
- GS Bucket: msml640-dataset
- Region: us-central1

## Setup Steps

### 1. Install Google Cloud SDK

```bash
brew install google-cloud-sdk
gcloud auth login
gcloud config set project msml640
```

### 2. Install Python Dependencies

```bash
pip install google-cloud-aiplatform google-cloud-storage
```

### 3. Reuse Object Detection Docker Image

We can reuse the Object Detection Docker image from the Object Detection training guide.

### 4. Prepare Your Dataset

```bash
export PROJECT_ID="msml640"
export REGION="us-central1"
export BUCKET_NAME="msml640-dataset"
export REPOSITORY="vertex-ai"
export IMAGE_NAME="yolo11-rubiks-cube"
export IMAGE_TAG="latest"
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME/
gsutil -m cp -r "synthetic-data/resized" gs://$BUCKET_NAME/color-detection/dataset/synthetic-data/
gsutil -m cp -r "synthetic-data/labels/labels_numeric.csv" gs://$BUCKET_NAME/color-detection/dataset/synthetic-data/labels/labels_numeric.csv

gsutil cp train_cnn.py gs://$BUCKET_NAME/color-detection/train_cnn.py 
```

### 5. Submit Training Job

The `config.yaml` file is already configured to use the custom Docker image. Simply submit:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=rubiks-cube-yolo11 \
  --config=config.yaml
```

## Downloading Trained Model

After training completes, download your model:

```bash
gsutil cp gs://$BUCKET_NAME/color-detection/models/cube_cnn.pt ./cube_cnn.pt
```
