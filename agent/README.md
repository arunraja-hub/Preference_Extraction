# To run locally 💻

`python train.py --root_dir some/dir`

`python eval.py --root_dir some/dir`

You can run both of these at the same time, or `train.py` followed by `eval.py`.

To track the progress, you can launch Tensorboard: `tensorboard --logdir some/dir` to see the result.

You probably won't need to modify any of the training code, since it can all be configured by the files in the configs directory.
The training and environment parameters can be changed there. See https://github.com/google/gin-config to learn more.

Also see https://www.tensorflow.org/agents

# To launch on Google Cloud ☁️
This process will download a few GB the first time. It will be faster the next time.

Follow instructions at the [Before you begin section](https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin) on Google Cloud's website.
    
    chmod +x launch_cloud.sh
    ./launch_cloud.sh job_name env_type agent_type hyperp_tune_bool

If the dependencies change, you'll need to re push the base docker container. This will take a lot of time and you'll need to upload a few gigs.
To re push the base docker image:

Modify the DockerfileBase file.
    
    ENV_TYPE=doom
    BASE_IMAGE_URI=gcr.io/preference-extraction/pref_extract_base
    docker build -f configs/$ENV_TYPE/DockerfileBase -t $BASE_IMAGE_URI ./
    docker push $BASE_IMAGE_URI

Note, `pref_extract_base` works for the Doom enviroment, for the PySC2 enviroment change the dependencies in `pref_extract_pysc2`

# To export an agent (locally)

`python export.py --root_dir some/dir --gin_file some/gin/config/file`

Additional options

* `--checkpoint` - Checkpoint at which export agent
* `--collect_data` - Number of experience data points to collect using this agent
