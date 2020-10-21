# Preference Extractors

This folder contains all the script to run preference extraction techniques on pre-trained agents and baselines.

The pipeline it uses is similar to that for [training agents][../agent/]

## Types of extractors
1. Tensorflow Extractors (`tf_extractor.py`)
    a) CNN form observations (`cnn_from_obs`)
    b) Agent (`agent_extractor`)
   
2. PyTorch Extractors (`torch_extractor.py`)
    a) Subnetworks finder (`TorchExtractor`)

## Run locally

* Choose an extractor (torch or tf)
* Edit the gin configuration file
* Run `python3 extract_preferences.py --gin_file <PATH_TO_GIN_FILE>`

## Lunch on g cloud

* CNN from observations: `./launch_cloud.sh <JOB_NAME> tf 1 base`
* TF Agent: `./launch_cloud.sh <JOB_NAME> tf 1`
* PyTorch subnetworks finder: `./launch_cloud.sh <JOB_NAME> torch 1`

Note: the `1` in the command line signals to the platform to run hyperparameter tuning, if you want to simply run a job on the cloud without tuning replace the `1` with a `0`

## Modify dependencies

* Modify `DockerfileBase`
* Then in command line
```
    BASE_IMAGE_URI=gcr.io/preference-extraction/pref_extract_tf_torch
    docker build -f DockerfileBase -t $BASE_IMAGE_URI ./
    docker push $BASE_IMAGE_URI
```