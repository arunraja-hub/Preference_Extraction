To run:

`python train.py --root_dir some/dir`

`python eval.py --root_dir some/dir`

You can run both of these at the same time, or eval after train.

`tensorboard --logdir some/dir` to see the result.

You probably won't need to modify any of the training code, since it can all be configured by the files in the configs directory.
The training and environment parameters can be changed there. See https://github.com/google/gin-config to learn more.

Also see https://www.tensorflow.org/agents