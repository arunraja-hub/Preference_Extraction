import time

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

import numpy as np

from sklearn.utils import shuffle

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

class SlowlyUnfreezing(tf.keras.callbacks.Callback):
    """A callback to slowly unfreeze previously frozen activation layers"""
    def on_train_begin(self, logs):
        self.num_epochs = 0
        
    def on_epoch_end(self, epoch, logs):
        self.num_epochs += 1
        if self.num_epochs >= 50:
            self.model.layers[3].trainable = True
        if self.num_epochs >= 100:
            self.model.layers[2].trainable = True
        if self.num_epochs >= 150:
            self.model.layers[1].trainable = True
        if self.num_epochs >= 200:
            self.model.layers[0].trainable = True

@gin.configurable
def cnn_from_obs(input_shape, reg_amount, drop_rate):
    """
       Simple Convolutional Neural Network
       that extracts preferences from observations
    """

    model = tf.keras.models.Sequential([
        # This layer gets one of the color channels. It works better than using all of them.
        tf.keras.layers.Lambda(lambda x:  tf.expand_dims(x[:,:,:,tf.random.uniform((), 0,4, tf.int32)], 3),
                               input_shape=input_shape[1:]),
        tf.keras.layers.Conv2D(64, 2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_amount)),
        tf.keras.layers.Conv2D(32, 1, activation='relu', strides=1,
                               kernel_regularizer=tf.keras.regularizers.l2(reg_amount)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(reg_amount))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(.01), loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])

    return model

def reset_model_weights(model):    
    for keras_layer in model.layers:
        initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None)
        if len(keras_layer.weights) > 0:
            weights = initializer(shape=keras_layer.weights[0].shape)
            biases = initializer(shape=keras_layer.weights[1].shape)
            keras_layer.set_weights([weights, biases])

@gin.configurable            
def agent_extractor(agent_path, agent_last_layer, agent_freezed_layers, 
                    layer_sizes, reg_amount, drop_rate, randomize_weights):
    """
        Builds a network to extract preferences
        From the RL agent originally trained in the enviroment
    """
    agent = tf.keras.models.load_model(agent_path)
    layers = []
    for ix, layer_size in enumerate(layer_sizes):
        layers.append(tf.keras.layers.Dense(layer_size, activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(reg_amount), name='post_agent_{}'.format(ix)))
        if drop_rate > 0:
            layers.append(tf.keras.layers.Dropout(drop_rate))
            
    for ix, _ in enumerate(agent.layers[:agent_last_layer]):
        if ix in agent_freezed_layers:
            agent.layers[ix].trainable = False
        else:
            agent.layers[ix].trainable = True
    
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
        
    def train_best_logs(self, xs, ys, do_summary=True):
        """
            Trains the model and retruns the logs of the best epoch.
            Randomly splits the train and val data before training.
        """
        
        tf.keras.backend.clear_session()
        
        xs, ys = shuffle(xs, ys)
        xs_val = xs[self.num_train:self.num_train+self.num_val]
        ys_val = ys[self.num_train:self.num_train+self.num_val]
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0)
        best_stats = BestStats()
        callbacks = [early_stopping, best_stats]
        if self.slowly_unfreezing:
            callbacks += [SlowlyUnfreezing()]

        model = self.extractor_fn()
        model.fit(xs[:self.num_train], ys[:self.num_train], epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=callbacks, validation_data=(xs_val, ys_val), verbose=0)

        if do_summary:
            model.summary()
            print("best train accuracy:", best_stats.bestTrain)
            print("Number of epochs:", best_stats.num_epochs)
        
        return best_stats.bestLogs
    
    def multiple_train_ave(self, xs, ys, do_summary = True):
        """
            Trains the model multiple times with the same parameters and returns the average metrics
        """
        
        start = time.time()
        all_val_auc = []
        all_val_accuracy = []

        
        for i in range(self.num_repeat):
            logs = self.train_best_logs(xs, ys, do_summary=do_summary)
            all_val_auc.append(get_val_auc(logs))
            all_val_accuracy.append(logs.get('val_accuracy'))
            do_summary = False 

        mean_val_auc = np.mean(all_val_auc)
        mean_val_accuracy = np.mean(all_val_accuracy)
        metric = (mean_val_auc + mean_val_accuracy) / 2.0
        print_data = ("mean_val_auc", mean_val_auc, "mean_val_accuracy", mean_val_accuracy, 
                      "metric", metric, "val_auc_std", np.std(all_val_auc), "val_accuracy_std", np.std(all_val_accuracy))

        end = time.time()
        print("Seconds per hyperparam config", end - start)

        return metric, print_data
    
    def train(self, xs, ys, do_summary = True):
        
        best_metric = -float('inf')
        run_num = 0
        metric, print_data = self.multiple_train_ave(xs, ys, do_summary)
        print(print_data)