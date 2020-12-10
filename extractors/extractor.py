import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

import numpy as np

import hypertune

@gin.configurable
class Extractor(object):
    def __init__(self, num_train, num_val, num_repeat = 5):
        self.num_repeat = num_repeat
        self.num_train = num_train
        self.num_val = num_val

    def train_single_shuffle(self, xs, ys, do_summary):
        """
            Trains the model and reruns the logs of the best epoch.
            Randomly splits the train and val data before training.
        """

        randomize = np.arange(len(xs))
        np.random.shuffle(randomize)
        xs = xs[randomize]
        ys = ys[randomize]

        xs_train = xs[:self.num_train]
        ys_train = ys[:self.num_train]
        xs_val = xs[self.num_train:self.num_train + self.num_val]
        ys_val = ys[self.num_train:self.num_train + self.num_val]

        return self.train_single(xs_train, ys_train, xs_val, ys_val, do_summary)

    @gin.configurable
    def train(self, xs, ys, do_summary):
        """
            Trains the model multiple times with the same parameters and returns the average metrics
        """

        all_val_auc = []
        all_val_accuracy = []

        for i in range(self.num_repeat):
            single_train_metrics = self.train_single_shuffle(xs, ys, do_summary=do_summary and (i == 0))
            all_val_auc.append(single_train_metrics['val_auc'])
            all_val_accuracy.append(single_train_metrics['val_accuracy'])

        metrics = {
            "mean_val_auc": np.mean(all_val_auc),
            "mean_val_accuracy": np.mean(all_val_accuracy),
            "val_auc_std": np.std(all_val_auc),
            "val_accuracy_std": np.std(all_val_accuracy)
        }

        if do_summary:
            print(metrics, flush=True)

        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='mean_val_auc',
            metric_value=metrics['mean_val_auc'])

        return metrics