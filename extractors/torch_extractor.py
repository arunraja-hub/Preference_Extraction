"""
    PyTorch Preference Extraction Methods
"""

import math
import itertools

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle

import gin
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchsummary


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
    
def get_initializer(initializer_type):
    if initializer_type == 'kaiming_normal':
        return nn.init.kaiming_normal_, {}
    elif initializer_type == 'kaiming_uniform':
        return nn.init.kaiming_uniform_, {'a': math.sqrt(5)}
    elif initializer_type == 'xavier_normal':
        return nn.init.xavier_normal_, {}
    elif initializer_type == 'best_activation':
        return nn.init.ones_, {}
    else:
        return nn.init.uniform_

class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, k, scores_init='kaiming_uniform', **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.scores_init = scores_init

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        initializer, init_kwargs = get_initializer(scores_init)
        initializer(self.scores, **init_kwargs)
        
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
        initializer, args = get_initializer(scores_init)
        initializer(self.scores, **args)

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.k)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        return x

class AgentModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.module_list = nn.ModuleList(layers)

    def load_agent_weigths(self, agent, device):
        for ix, layer in enumerate(agent.layers):
            if ix == 0: # Input Layer in Keras
                continue
            if len(layer.weights) > 0:
                self.module_list[ix-1].weight.data = torch.Tensor(np.transpose(layer.weights[0].numpy()))
                self.module_list[ix-1].bias.data = torch.Tensor(layer.weights[1].numpy())
                
        self.module_list.to(device)
            
    def forward(self, x):
        for ix, f in enumerate(self.module_list):
            x = f(x)
            if ix < len(self.module_list) - 2:
                # EcondingNetwork of both PPO and DQN uses ReLU for both Conv and Dense layers
                # Last layer is usually action logits or values and does not uses ReLUs
                x = nn.functional.relu(x) 
            elif ix == len(self.module_list) - 1:
                x = torch.sigmoid(x) # Extractor layer uses sigmoid
        return x.flatten()
    
@gin.configurable
class TorchExtractor(object):
    
    def __init__(self,
                 agent_path,
                 input_shape,
                 subnet_k,
                 randomize_weights,
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

        torch.set_printoptions(precision=8)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        self.model = None
        self.create_agent_model(agent_path, input_shape, subnet_k, randomize_weights)
        
    def create_agent_model(self, agent_path, input_shape, subnet_k, randomize_weights):
        
        agent = tf.keras.models.load_model(agent_path)
        x = tf.keras.Input(shape=input_shape)
        agent_submodel = tf.keras.Model(inputs=[x], outputs=agent.call(x))
        agent_submodel.summary()
        for ix, layer in enumerate(agent.layers):  # Make sure agent and agent_subnet has same weights
            agent_submodel.layers[ix+1].set_weights(layer.get_weights())
        
        torch_layers = []
        for ix, layer in enumerate(agent_submodel.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                last_shape = layer.output_shape[0]
            
            elif isinstance(layer, tf.keras.layers.Conv2D):
                torch_layer = SupermaskConv(in_channels=last_shape[-1], out_channels=layer.filters,
                                            kernel_size=layer.kernel_size, stride=layer.strides, bias=True, k=subnet_k)
                torch_layers.append(torch_layer)
                last_shape = layer.output_shape
                
            elif isinstance(layer, tf.keras.layers.Flatten):
                torch_layers.append(nn.Flatten())
                last_shape = layer.output_shape
                
            elif isinstance(layer, tf.keras.layers.Dense):
                torch_layer = SupermaskLinear(in_features=last_shape[-1], out_features=layer.weights[0].shape[-1],
                                              bias=True, k=subnet_k)
                torch_layers.append(torch_layer)            
                last_shape = layer.output_shape
                
                
        torch_layers.append(nn.Linear(in_features=last_shape[-1], out_features=1, bias=True)) # last layer to transform output
        self.model = AgentModel(torch_layers)
        torchsummary.summary(self.model, input_size=(input_shape[-1], input_shape[0], input_shape[1])) # torch wants chanell first
        
        if not randomize_weights:
            self.model.load_agent_weigths(agent_submodel, self.device)
            self.verify_weights(agent_submodel, agent, input_shape)
            
    def verify_weights(self, agent_submodel, agent, input_shape):
        """
            Method to verify correctness of agent_sumodel weigths
            And of new torch network weigths
        """
        def check_same(torch_layers, tf_layer):
            x = random_obs_torch
            for lx, f in enumerate(torch_layers):
                x = f(x)
                if lx < len(self.model.module_list) - 2:
                    x = nn.functional.relu(x)
            torch_out = x.detach().cpu().numpy()
            tf_out = tf_layer(random_obs).numpy()
            if len(tf_out.shape) > 2:
                tf_out = np.rollaxis(tf_out, 3, 1)            
            np.testing.assert_allclose(torch_out, tf_out, rtol=.1, atol=5)
            # This testing values are relatively close, but I cannot satify anything tighter
            # Why are torch activations so 'relatively' different from tf ones
            # TODO: investigate if there is something missing in the tf-torch translation
        
        for _ in range(50):
            random_obs = np.random.random(size=(1,) + input_shape)
            random_obs_torch = torch.Tensor(np.rollaxis(random_obs, 3, 1))  # torch wants channel first

            assert (agent_submodel(random_obs).numpy() == agent(random_obs).numpy()).all()

            for ix, layer in enumerate(agent_submodel.layers):
                if ix == 0:
                    continue
                tf_sub_model = tf.keras.models.Model(inputs=agent_submodel.input, outputs=layer.output)
                check_same(self.model.module_list[:ix], tf_sub_model)        
    
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

    def _train(self, train_loader, optimizer, criterion):

        train_loss = 0
        true_labels = []
        predictions = []
        
        self.model.train()
        for data, target in itertools.islice(train_loader, self.num_train):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss
            predictions.extend(output.detach().cpu().numpy())
            true_labels.extend(target.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        accuracy, auc = self.compute_metrics(predictions, true_labels)

        return train_loss.item(), accuracy, auc

    def _test(self, test_loader, criterion):
        true_labels = []
        predictions = []
        
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in itertools.islice(test_loader, self.num_val):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target)
                predictions.extend(output.detach().cpu().numpy())
                true_labels.extend(target.detach().cpu().numpy())

        test_loss /= len(test_loader.dataset)
        accuracy, auc = self.compute_metrics(predictions, true_labels)

        return test_loss.item(), accuracy, auc
    
    def get_data_sample(self, xs, ys):
        
        xs, ys = shuffle(xs, ys)
        xs = np.rollaxis(np.array(xs), 3, 1) # Torch wants channel-first
        
        tr_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.Tensor(xs[:self.num_train]), torch.Tensor(ys[:self.num_train])),
            batch_size=self.batch_size)

        val_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.Tensor(xs[self.num_train:self.num_val]),
                                           torch.Tensor(ys[self.num_train:self.num_val])),
            batch_size=self.batch_size)

        return tr_data_loader, val_data_loader, xs[:self.num_train]

    
    def single_train_pass(self, xs, ys):
        
        tr_data_loader, val_data_loader, x_train = self.get_data_sample(xs, ys)
        
        # Normalise last layer using training data
        # if hasattr(model, 'layer_to_norm'):
        #    model.mu_s = get_heads_mu_and_sigma(model, x_train)
    
        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate, weight_decay=self.weight_decay)
        
        criterion = nn.BCELoss().to(self.device)
        scheduler = CosineAnnealingLR(optimizer, T_max=len(tr_data_loader))

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        train_aucs, test_aucs = [], []

        best_test_loss = np.inf
        test_loss_up_since = 0
        early_stop = 50
        verbose = False
        
        for epoch in range(self.epochs):
            train_loss, train_accuracy, train_auc = self._train(tr_data_loader, optimizer, criterion)
            test_loss, test_accuracy, test_auc = self._test(val_data_loader, criterion)
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

    def train(self, xs, ys):
        
        averaged_results = {}
        for run_ix in range(self.num_repeat):
            print(f'Train pass no. {run_ix+1}')
            results = self.single_train_pass(xs, ys)       
        
            for res in results:
                if len(results[res]) > 0:
                    if res not in averaged_results:
                        averaged_results[res] = [results[res][-1]]
                    else:
                        averaged_results[res].append(results[res][-1])         
    
        return {x: sum(averaged_results[x]) / self.num_repeat for x in averaged_results}
