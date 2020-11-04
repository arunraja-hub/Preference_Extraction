import time

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

import numpy as np

import hypertune

import matplotlib.pyplot as plt
import random

def get_val_auc(logs):
    for key in logs:
        if key.startswith('val_auc'):
            return logs[key]

class BestStats(tf.keras.callbacks.Callback):
    """A callback to keep track of the best val accuracy and auc seen so far."""
    def on_train_begin(self, logs):
        self.bestMetric = -float('inf')
        self.bestLogs = None
        self.bestTrain = -float('inf')
        self.num_epochs = 0

    def on_epoch_end(self, epoch, logs):
        self.num_epochs += 1
        self.bestTrain = max(self.bestTrain, logs.get('accuracy'))

        val_accuracy = logs.get('val_accuracy')
        if val_accuracy == None:
            return 
        
        val_auc = get_val_auc(logs)
        metric = (val_accuracy + val_auc) / 2.0
        
        if metric > self.bestMetric:
            self.bestMetric = metric
            self.bestLogs = logs

@gin.configurable
class SlowlyUnfreezing(tf.keras.callbacks.Callback):
    """A callback to slowly unfreeze previously frozen activation layers"""
    def __init__(self, unfreze_every_n_epochs, start_unfreezing_from = 3):
        self.unfreze_every_n_epochs = unfreze_every_n_epochs
        self.start_unfreezing_from = start_unfreezing_from
        
    def on_train_begin(self, logs):
        self.num_epochs = 0
        self.start_unfreezing_from = min(self.start_unfreezing_from, len(self.model.layers) - 1)
        for ix in range(self.start_unfreezing_from, -1, -1):
            self.model.layers[ix].trainable = False
            
    def on_epoch_end(self, epoch, logs):
        self.num_epochs += 1
        layers_to_unfreeze = int(self.num_epochs / self.unfreze_every_n_epochs)
        for ix in range(self.start_unfreezing_from, self.start_unfreezing_from - layers_to_unfreeze, -1):
            if ix >= 0:
                self.model.layers[ix].trainable = True

def get_layer_sizes(first_size, last_size, num_layers):
    return np.linspace(first_size, last_size, num_layers, dtype=np.int32)

def get_dense_layers(fc_layer_sizes, reg_amount, drop_rate):
    layers = []
    for layer_size in fc_layer_sizes:
        layers.append(tf.keras.layers.Dense(layer_size, activation='relu',
                                              kernel_regularizer=tf.keras.regularizers.l2(reg_amount)))
        layers.append(tf.keras.layers.Dropout(drop_rate))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(reg_amount)))

    return layers

@gin.configurable
def cnn_from_obs(input_shape, cnn_first_size, cnn_last_size, cnn_num_layers, cnn_stride_every_n,
                 fc_first_size, fc_last_size, fc_num_layers, reg_amount, drop_rate, learning_rate, pick_random_col_ch=False, pooling=False):
    """
       Simple Convolutional Neural Network
       that extracts preferences from observations
    """
    layers = []
    if pick_random_col_ch:
         # layer to get one of the color channels. It works better than using all of them in the gridworld
        layers.append(tf.keras.layers.Lambda(lambda x: tf.expand_dims(
            x[:,:,:,tf.random.uniform((), 0,4, tf.int32)], 3), input_shape=input_shape))

    conv_layer_sizes = get_layer_sizes(cnn_first_size, cnn_last_size, cnn_num_layers)
    for i, layer_size in enumerate(conv_layer_sizes):
        if ((i+1) % cnn_stride_every_n) == 0:
            stride = 2
        else:
            stride = 1
        layers.append(tf.keras.layers.Conv2D(layer_size, 3, strides=stride, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(reg_amount)))

    if pooling:
        layers.append(tf.keras.layers.GlobalAveragePooling2D())

    layers.append(tf.keras.layers.Flatten())

    fc_layer_sizes = get_layer_sizes(fc_first_size, fc_last_size, fc_num_layers)
    layers.extend(get_dense_layers(fc_layer_sizes, reg_amount, drop_rate))

    model = tf.keras.models.Sequential(layers)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model

def reset_model_weights(model):
    for keras_layer in model.layers:
        if len(keras_layer.weights) > 0:
            weights = keras_layer.kernel_initializer(shape=keras_layer.weights[0].shape)
            biases = keras_layer.bias_initializer(shape=keras_layer.weights[1].shape)
            keras_layer.set_weights([weights, biases])

def print_data(xs, ys):
    """This function can be used to double check the data."""
    for _ in range(10):
        i = random.randint(0, len(xs)-1)
        print(ys[i])
        plt.imshow(xs[i,:,:,:3])
        plt.show()
    print("="*10)

@gin.configurable            
def agent_extractor(agent_path, agent_last_layer, agent_freezed_layers, 
                    fc_layer_sizes, reg_amount, drop_rate, randomize_weights):
    """
        Builds a network to extract preferences
        From the RL agent originally trained in the enviroment
    """
    agent = tf.keras.models.load_model(agent_path)
    for ix, _ in enumerate(agent.layers[:agent_last_layer]):
        agent.layers[ix].trainable = ix not in agent_freezed_layers

    layers = get_dense_layers(fc_layer_sizes, reg_amount, drop_rate)

    model = tf.keras.models.Sequential(agent.layers[:agent_last_layer] + layers + [
        tf.keras.layers.Dense(1, activation='sigmoid', 
        kernel_regularizer=tf.keras.regularizers.l2(reg_amount), name='output')])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(.01), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    
    if randomize_weights:
        reset_model_weights(model)
    
    return model

@gin.configurable
class TfExtractor(object):
    
    def __init__(self,
                 extractor_fn,
                 num_train,
                 num_val,
                 num_repeat = 5,
                 slowly_unfreezing = False,
                 epochs = 500,
                 batch_size = 128):
        
        self.extractor_fn = extractor_fn
        self.num_train = num_train
        self.num_val = num_val
        self.num_repeat = num_repeat
        self.epochs = epochs
        self.batch_size = batch_size
        self.slowly_unfreezing = slowly_unfreezing
        
    def train_single_shuffle(self, xs, ys, do_summary):
        """
            Trains the model and retruns the logs of the best epoch.
            Randomly splits the train and val data before training.
        """
        
        randomize = np.arange(len(xs))
        np.random.shuffle(randomize)
        xs = xs[randomize]
        ys = ys[randomize]

        xs_train = xs[:self.num_train]
        ys_train = ys[:self.num_train]
        xs_val = xs[self.num_train:self.num_train+self.num_val]
        ys_val = ys[self.num_train:self.num_train+self.num_val]

        return self.train_single(xs_train, ys_train, xs_val, ys_val, do_summary)

    def train_single(self, xs_train, ys_train, xs_val, ys_val, do_summary):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0)
        best_stats = BestStats()
        callbacks = [early_stopping, best_stats]
        if self.slowly_unfreezing:
            callbacks += [SlowlyUnfreezing()]

        model = self.extractor_fn()
        model.fit(xs_train, ys_train, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=callbacks, validation_data=(xs_val, ys_val), verbose=0)

        if do_summary:
            model.summary()
            print("best train accuracy:", best_stats.bestTrain)
            print("Number of epochs:", best_stats.num_epochs, flush=True)

        return {'val_auc': get_val_auc(best_stats.bestLogs), 'val_accuracy': best_stats.bestLogs.get('val_accuracy')}

    @gin.configurable
    def train(self, xs, ys, do_summary):
        """
            Trains the model multiple times with the same parameters and returns the average metrics
        """
        
        all_val_auc = []
        all_val_accuracy = []

        for i in range(self.num_repeat):
            single_train_metrics = self.train_single_shuffle(xs, ys, do_summary=do_summary and (i==0))
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