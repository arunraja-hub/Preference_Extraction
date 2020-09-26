# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh job_name 0
# ./launch_cloud.sh job_name 1 for hparam tune.

# This launches a new training on google cloud.
# If you want to do something else, you can use these commands as examples.

JOB_NAME=$1_$(($(date +%s)-1600400000)) ;
echo launching $JOB_NAME

BUCKET_ID=pref_extract_train_output
PROJECT_ID=preference-extraction
IMAGE_URI=gcr.io/$PROJECT_ID/pref_extract_train_container:$JOB_NAME
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

docker build -f Dockerfile -t $IMAGE_URI ./
echo "Container built. You can test localy with"
echo "docker run $IMAGE_URI --job-dir ~/pref_ext_train/$JOB_NAME"

echo "Pushing and launching"
docker push $IMAGE_URI

if [ $2 -eq 1 ]
then
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region us-central1 \
    --master-image-uri $IMAGE_URI \
    --scale-tier BASIC \
    --job-dir $JOB_DIR \
    --config hptuning_config.yaml
else
  gcloud beta ai-platform jobs submit training $JOB_NAME \
    --region us-central1 \
    --master-image-uri $IMAGE_URI \
    --scale-tier BASIC \
    --job-dir $JOB_DIR
fi

echo "See the job training here: https://console.cloud.google.com/ai-platform/jobs?authuser=1&project=preference-extraction"

echo "python3 -m tensorboard.main --logdir $JOB_DIR"