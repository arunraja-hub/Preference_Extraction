# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh baseline tf.gin hptuning_config_tf_baseline.yaml
# ./launch_cloud.sh subnet torch.gin hptuning_config_torch.yaml
# ./launch_cloud.sh agent_extract tf.gin hptuning_config_tf.yaml
# for hparam tune.

# This launches a new training on google cloud.
# If you want to do something else, you can use these commands as examples.

JOB_NAME=$1_$(($(date +%s)-1600400000)) ;
echo launching $JOB_NAME

BUCKET_ID=pref_extract_train_output
PROJECT_ID=preference-extraction
IMAGE_URI=gcr.io/$PROJECT_ID/pref_extract_tf_torch:$JOB_NAME
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME

docker build -f Dockerfile -t $IMAGE_URI ./

echo "Pushing and launching"
docker push $IMAGE_URI

echo "Container built. You can test localy with"
echo "docker run $IMAGE_URI $GIN_CONFIG"

CLOUD_CONFIG="--config configs/$3"
GIN_CONFIG="--gin_file configs/$2"

gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region us-central1 \
  --master-image-uri $IMAGE_URI \
  $CLOUD_CONFIG \
  -- $GIN_CONFIG

echo "See the job training here: https://console.cloud.google.com/ai-platform/jobs?authuser=1&project=preference-extraction"