# Usage:
# Follow instructions at the Before you begin section of https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin
# chmod +x launch_cloud.sh
# ./launch_cloud.sh name tf.gin hptuning_config_tf_baseline.yaml
# ./launch_cloud.sh name torch.gin hptuning_config_torch.yaml
# ./launch_cloud.sh name torch.gin hptuning_config_torch_baseline.yaml
# ./launch_cloud.sh name tf.gin hptuning_config_tf.yaml
# for hparam tune.

# This launches a new training on google cloud.
# If you want to do something else, you can use these commands as examples.

CONFIG_NO_EXT="$(basename $3 .yaml)"
JOB_NAME="${CONFIG_NO_EXT}_$1_"$(($(date +%s)-1601800000)) ;
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

# Regions: asia-east1 asia-east2 asia-northeast1 asia-northeast2 asia-northeast3 asia-south1 asia-southeast1 asia-southeast2 australia-southeast1
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region asia-south1 \
  --master-image-uri $IMAGE_URI \
  $CLOUD_CONFIG \
  -- $GIN_CONFIG

echo "See the job training here: https://console.cloud.google.com/ai-platform/jobs?authuser=1&project=preference-extraction"
