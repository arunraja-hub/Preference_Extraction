"""
    PyTorch Preference Extraction Methods
"""

import math
import itertools

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

import gin
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR

class GetSubnet(autograd.Function):
    """
        Original code from 'What's hidden in a randomly weighted neural network?'
        Implemented at https://github.com/allenai/hidden-networks
    """
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, k, scores_init='kaiming_uniform', **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.scores_init = scores_init

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.scores_init == 'kaiming_normal':
            nn.init.kaiming_normal_(self.scores)
        elif self.scores_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        elif self.scores_init == 'xavier_normal':
            nn.init.xavier_normal_(self.scores)
        elif self.scores_init == 'best_activation':
            nn.init.ones_(self.scores)
        else:
            nn.init.uniform_(self.scores)
        
        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.k)
        w = self.weight * subnet
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, *args, k, scores_init='kaiming_uniform', **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.scores_init = scores_init

        # initialize the scores and weights
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.scores_init == 'kaiming_normal':
            nn.init.kaiming_normal_(self.scores)
        elif self.scores_init == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        elif self.scores_init == 'xavier_normal':
            nn.init.xavier_normal_(self.scores)
        elif self.scores_init == 'best_activation':
            nn.init.ones_(self.scores)
        else:
            nn.init.uniform_(self.scores)

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.k)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        return x

@gin.configurable
class TorchExtractor(object):
    
    def __init__(self,
                 agent_path,
                 input_shape,
                 subnet_k,
                 num_train,
                 num_val,
                 num_repeat = 5,
                 epochs = 500,
                 batch_size = 128,
                 learning_rate = 1e-2,
                 weight_decay = 0):
        
        self.num_train = num_train
        self.num_val = num_val
        self.num_repeat = num_repeat
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.create_agent_model(agent_path, input_shape, subnet_k)

    
    def create_agent_model(self, agent_path, input_shape, subnet_k):
        
        agent = tf.keras.models.load_model(agent_path)

        torch_layers = []
        last_shape = input_shape
        for ix, layer in enumerate(agent.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                torch_layer = SupermaskConv(in_channels=last_shape[-1], out_channels=layer.filters,
                                            kernel_size=layer.kernel_size, stride=layer.strides, bias=True, k=subnet_k)
                torch_layers.append(torch_layer)
                last_shape = layer.weights[0].shape
            elif isinstance(layer, tf.keras.layers.Flatten):
                torch_layers.append(nn.Flatten())
                if ix == 0:  # first flatten layer does not use batch size
                    last_shape = [np.prod(last_shape[1:])]
                else:
                    last_shape = [np.prod(last_shape)]
            elif isinstance(layer, tf.keras.layers.Dense):
                torch_layer = SupermaskLinear(in_features=last_shape[-1], out_features=layer.weights[0].shape[-1],
                                              bias=True, k=subnet_k)
                last_shape = layer.weights[0].shape
                torch_layers.append(torch_layer)
        
        torch_model = nn.ModuleList(torch_layers)
        print(torch_model)
            
    """
       Train/Test function for Randomly Weighted Hidden Neural Networks Techniques
       Adapted from https://github.com/NesterukSergey/hidden-networks/blob/master/demos/mnist.ipynb
    """
    @staticmethod
    def compute_metrics(predictions, true_labels):
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        accuracy = np.sum(np.equal((predictions > 0.5).astype(int), true_labels)) / len(true_labels)
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
        auc = metrics.auc(fpr, tpr)
        return accuracy, auc

    def _train(self, model, device, train_loader, optimizer, criterion):

        train_loss = 0
        true_labels = []
        predictions = []

        model.train()

        for data, target in itertools.islice(train_loader, self.num_train):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss
            predictions.extend(output.detach().cpu().numpy())
            true_labels.extend(target.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        accuracy, auc = compute_metrics(predictions, true_labels)

        return train_loss.item(), accuracy, auc

    def _test(self, model, device, test_loader, criterion):
        true_labels = []
        predictions = []

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in itertools.islice(test_loader, self.num_val):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target)
                predictions.extend(output.detach().cpu().numpy())
                true_labels.extend(target.detach().cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy, auc = compute_metrics(predictions, true_labels)

        return test_loss.item(), accuracy, auc
    
    def single_train_pass(self, model):
        
        tr_data_loader, val_data_loader, x_train = get_data_sample(xs, ys)
        
        # Normalise last layer using training data
        if hasattr(model, 'layer_to_norm'):
            model.mu_s = get_heads_mu_and_sigma(model, x_train)
    
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.learning_rate, weight_decay=self.weight_decay)
        
        criterion = nn.BCELoss().to(device)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(tr_data_loader))

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        train_aucs, test_aucs = [], []

        best_test_loss = np.inf
        test_loss_up_since = 0
        early_stop = 100
        verbose = False
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy, train_auc = self._train(model, device, tr_data_loader, optimizer, criterion)
            test_loss, test_accuracy, test_auc = self._test(model, device, val_data_loader, criterion)
            scheduler.step()
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                test_loss_up_since = 0
            test_loss_up_since += 1
            if test_loss_up_since > early_stop:
                print('Epoch - ', epoch, 'Early stopping')
                break
            if verbose:
                print('Epoch - ', epoch)
                print('Train metrics: loss', train_loss, 'accuracy', train_accuracy, 'auc', train_auc)
                print('Val metrics: loss', test_loss, 'accuracy', test_accuracy, 'auc', test_auc)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_accuracy)
            train_aucs.append(train_auc)
            test_accs.append(test_accuracy)
            test_aucs.append(test_auc)

        return {'trainLoss': train_losses, 'testLoss': test_losses, 
                'trainAccuracy': train_accs, 'testAccuracy': test_accs,
                'trainAUC': train_aucs, 'testAUC': test_aucs}

    def train(self, model):
        
        averaged_results = {}
        for run_ix in range(self.num_run):
            results = self.single_train_pass(model)       
        print(f'Train pass no. {run_ix+1}')

        if (run_ix == 0):
            print(model)
        
        for res in results:
            if len(results[res]) > 0:
                if res not in averaged_results:
                    averaged_results[res] = [results[res][-1]]
                else:
                    averaged_results[res].append(results[res][-1])         
    
        return {x: sum(averaged_results[x]) / self.num_run for x in averaged_results}
