# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh job_name

# This launches a new training on google cloud.
# If you want to do something else, you can use these commands as examples.

JOB_NAME=$1_$(($(date +%s)-1600400000)) ;
echo launching $JOB_NAME

BUCKET_ID=pref_extract_train_output
PROJECT_ID=preference-extraction
IMAGE_URI=gcr.io/$PROJECT_ID/pref_extract_train_container:$JOB_NAME

docker build -f Dockerfile -t $IMAGE_URI ./
echo "Container built. You can test localy with docker run $IMAGE_URI --root_dir your_root_dir"

echo "Pushing and launching"
docker push $IMAGE_URI

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region us-central1 \
  --master-image-uri $IMAGE_URI \
  --scale-tier BASIC \
  --config hptuning_config.yaml \
  -- \
  --root_dir=gs://$BUCKET_ID/$JOB_NAME

echo "See the job training here: https://console.cloud.google.com/ai-platform/jobs?authuser=1&project=preference-extraction"
