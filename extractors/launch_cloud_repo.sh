PRE_JOB_NAME=$1
TRIAL_NUM=$2
JOB_NAME="repo_${PRE_JOB_NAME}_trial${TRIAL_NUM}"$(($(date +%s)-1601800000)) ;
echo launching $JOB_NAME

BUCKET_ID=pref_extract_train_output
PROJECT_ID=preference-extraction
IMAGE_URI=gcr.io/$PROJECT_ID/pref_extract_tf_torch:$JOB_NAME
JOB_DIR=gs://$BUCKET_ID/$JOB_NAME
PREV_JOB_DIR=gs://$BUCKET_ID/$PRE_JOB_NAME/$TRIAL_NUM

docker build -f Dockerfile -t $IMAGE_URI ./

echo "Pushing and launching"
docker push $IMAGE_URI

GIN_CONFIG="--gin_file $PREV_JOB_DIR/operative_config-final.gin"

echo "JOB_NAME $JOB_NAME GIN_CONFIG $GIN_CONFIG"

# Regions: asia-east1 asia-east2 asia-northeast1 asia-northeast2 asia-northeast3 asia-south1 asia-southeast1 asia-southeast2 australia-southeast1
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --region asia-south1 \
  --master-image-uri $IMAGE_URI \
  --job-dir $JOB_DIR \
  $CLOUD_CONFIG \
  -- $GIN_CONFIG