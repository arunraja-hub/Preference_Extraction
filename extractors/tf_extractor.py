import time

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

import numpy as np

import extractor

def get_val_auc(logs):
    for key in logs:
        if key.startswith('val_auc'):
            return logs[key]

class LastStats(tf.keras.callbacks.Callback):
    """A callback to keep track of the best val accuracy and auc seen so far."""
    def on_train_begin(self, logs):
        self.lastAuc = -float('inf')
        self.lastLogs = None
        self.lastTrain = -float('inf')
        self.num_epochs = 0

    def on_epoch_end(self, epoch, logs):
        self.num_epochs += 1
        self.lastTrain = logs.get('accuracy')

        val_accuracy = logs.get('val_accuracy')
        if val_accuracy == None:
            return 
        
        self.lastAuc = get_val_auc(logs)
        self.lastLogs = logs

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

def get_dense_layers(fc_layer_sizes, reg_amount, drop_rate):
    layers = []
    for layer_size in fc_layer_sizes:
        layers.append(tf.keras.layers.Dense(layer_size, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(reg_amount),
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                            bias_initializer=tf.keras.initializers.RandomUniform()))
        layers.append(tf.keras.layers.Dropout(drop_rate))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(reg_amount)))

    return layers

def custom_loss():
    return tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, name='binary_crossentropy',
                                              reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

def cosine_anneal_lr(initial_lr, t_max):
    return tf.keras.experimental.CosineDecay(initial_lr, t_max)


@gin.configurable
def cnn_from_obs(input_shape, cnn_first_size, cnn_last_size, cnn_num_layers, cnn_stride_every_n, kernel_size,
                 fc_first_size, fc_last_size, fc_num_layers, reg_amount, drop_rate, learning_rate, cosine_anneal_t_max,
                 pick_random_col_ch, pooling):
    """
       Simple Convolutional Neural Network
       that extracts preferences from observations
    """
    print("TF cnn_from_obs")
    layers = []
    if pick_random_col_ch:
         # layer to get one of the color channels. It works better than using all of them in the gridworld
        layers.append(tf.keras.layers.Lambda(lambda x: tf.expand_dims(
            x[:,:,:,tf.random.uniform((), 0,4, tf.int32)], 3), input_shape=input_shape))

    conv_layer_sizes = extractor.get_layer_sizes(cnn_first_size, cnn_last_size, cnn_num_layers)
    for i, layer_size in enumerate(conv_layer_sizes):
        if ((i+1) % cnn_stride_every_n) == 0:
            stride = 2
        else:
            stride = 1
        layers.append(tf.keras.layers.Conv2D(layer_size, kernel_size, strides=stride, activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(reg_amount),
                                             kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                             bias_initializer=tf.keras.initializers.RandomUniform()))
        layers.append(tf.keras.layers.Dropout(drop_rate))

        

        
    if pooling:
        layers.append(tf.keras.layers.GlobalAveragePooling2D())

    layers.append(tf.keras.layers.Flatten())

    fc_layer_sizes = extractor.get_layer_sizes(fc_first_size, fc_last_size, fc_num_layers)
    layers.extend(get_dense_layers(fc_layer_sizes, reg_amount, drop_rate))

    model = tf.keras.models.Sequential(layers)

    model.compile(optimizer=tf.keras.optimizers.Adam(cosine_anneal_lr(learning_rate, cosine_anneal_t_max)), 
                  loss=custom_loss(), metrics=['accuracy', tf.keras.metrics.AUC()])

    return model

def reset_model_weights(model):
    for keras_layer in model.layers:
        if len(keras_layer.weights) > 0:
            weights = keras_layer.kernel_initializer(shape=keras_layer.weights[0].shape)
            biases = keras_layer.bias_initializer(shape=keras_layer.weights[1].shape)
            keras_layer.set_weights([weights, biases])

@gin.configurable            
def agent_extractor(agent_path, agent_last_layer, agent_freezed_layers, first_size, last_size, num_layers, 
                    reg_amount, drop_rate, learning_rate, cosine_anneal_t_max, randomize_weights):
    """
        Builds a network to extract preferences
        From the RL agent originally trained in the enviroment
    """
    print("TF agent_extractor")
    agent = tf.keras.models.load_model(agent_path)
    for ix, _ in enumerate(agent.layers[:agent_last_layer]):
        agent.layers[ix].trainable = ix not in agent_freezed_layers

    fc_layer_sizes = extractor.get_layer_sizes(first_size, last_size, num_layers)
    layers = get_dense_layers(fc_layer_sizes, reg_amount, drop_rate)

    model = tf.keras.models.Sequential(agent.layers[:agent_last_layer] + layers)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(cosine_anneal_lr(learning_rate, cosine_anneal_t_max)), 
                  loss=custom_loss(), metrics=['accuracy', tf.keras.metrics.AUC()])
    
    if randomize_weights:
        reset_model_weights(model)
    
    return model

@gin.configurable
class TfExtractor(extractor.Extractor):
    
    def __init__(self,
                 model_fn,
                 slowly_unfreezing,
                 epochs,
                 batch_size):
        super().__init__()
        print("Using TfExtractor", flush=True)

        self.model = model_fn()
        self.epochs = epochs
        self.batch_size = batch_size
        self.slowly_unfreezing = slowly_unfreezing

    def train_single(self, xs_train, ys_train, xs_val, ys_val):
        tf.keras.backend.clear_session()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=0)
        stats = LastStats()
        callbacks = [early_stopping, stats]
        if self.slowly_unfreezing:
            callbacks += [SlowlyUnfreezing()]

        self.model.fit(xs_train, ys_train, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=callbacks, validation_data=(xs_val, ys_val), verbose=0)

        self.model.summary()
        print("final train accuracy:", stats.lastTrain)
        print("Number of epochs:", stats.num_epochs, flush=True)

        return {'val_auc': get_val_auc(stats.lastLogs), 'val_accuracy': stats.lastLogs.get('val_accuracy')}
