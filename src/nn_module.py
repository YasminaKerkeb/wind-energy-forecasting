
import sklearn.metrics.cluster as cluster_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from matplotlib.pyplot import contourf
from matplotlib import cm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



class PyTorchNetwork(nn.Module):
    
    """
    Base class for Neural Network model.
        
    """
    
    def __init__(self,model, loss_fn, optimizer="Adam", n_epochs=3, n_batches= 1000, tolerance=1e-6,learning_rate=None):
        """
        Initialize a PyTorch neural network model given a loss function, optimizer and other parameters for training
        
        Params
        ------
        
        model:          A function defining the model using torch.Sequential
        loss_fn:        string representing the supervised learning problem among ("regression", "binary" or "multiclass")
        optimizer:      string representing the Optimizer algorithm (Exp: "Adam")
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned (default 3).
        n_epochs:       An integer specifying the maximum number of iterations
                        to do (default 10000)
        tolerance:      A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
                   

        """
        super().__init__()
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')    
        self.model=model
        model_func=lambda :self.model
        self.net=model_func()
        self.loss_fn=loss_fn
        self.n_batches=n_batches
        self.n_epochs=n_epochs
        self.tolerance=tolerance
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.optimization = self.optimizer_initializer()
        self.loss=self.loss_initializer()
        

    def forward(self,x):
        x = self.net(x)
        return x

    def optimizer_initializer(self):
        if self.optimizer and self.optimizer!='Adam':
            optimization = torch.optim.SGD(self.net.parameters(),learning_rate = self.learning_rate)
        else:
            optimization = torch.optim.Adam(self.net.parameters())
        return optimization

    def loss_initializer(self):
        if self.loss_fn=="regression":
            loss=torch.nn.MSELoss()
        elif self.loss_fn=="binary":
            loss=torch.nn.BCELoss()
        else:
            loss=torch.nn.CrossEntropyLoss()
        return loss

    @staticmethod
    def init_weights(m):
        #Use Xavier_uniform weight initialization
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def weight_initializer(self):
        #Use Xavier_uniform weight initialization
        self.model.apply(self.init_weights)
    

    def train(self,X,y):
        N,M=X.shape
        best_final_loss=1e100
        logging_frequency=self.n_batches/10
        batch_size=int(N/self.n_batches)
        # The dataloaders handle shuffling, batching, etc...
        data = TensorDataset(X, y)
        train_data = DataLoader(data, batch_size=batch_size)
        for r in range(self.n_epochs):
            print('\n\tEpoch: {}/{}'.format(r+1, self.n_epochs))
            # Initialize a model 
            
            #Initialize weights
            self.weight_initializer()

            # Train the network while displaying and storing the loss
            print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
            learning_curve = [] # setup storage for loss at each step
            old_loss = 1e6

            #Initialize the model
            for i,train_batch in enumerate(train_data):
                X_train, y_train=train_batch
                y_est = self.net(X_train) # forward propagation and predict labels on training set
                loss_eval= self.loss(y_est, y_train) # determine loss
                loss_value = loss_eval.data.numpy() #get numpy array instead of tensor
                learning_curve.append(loss_value) # save loss

                # Convergence check, see if the percentual loss decrease is within
                # tolerance:
                p_delta_loss = np.abs(loss_value-old_loss)/old_loss
                if p_delta_loss < self.tolerance: break
                old_loss = loss_value

                # display loss with some frequency:
                if (i != 0) & ((i+1) % logging_frequency == 0):
                    print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                    print(print_str)
                # do backpropagation of loss and optimize weights 
                self.optimization.zero_grad(); loss_eval.backward(); self.optimization.step()




            # display final loss
            print('\t\tFinal loss:')
            print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
            print(print_str)

            if loss_value < best_final_loss: 
                best_net = self.net
                best_final_loss = loss_value
                best_learning_curve = learning_curve
                
                # Return the best curve along with its final loss and learing curve
        return best_net, best_final_loss, best_learning_curve
        
    def validate(self,X_val,y_val):
        # Determine estimated class labels for validation set
        y_val_est = net(X_val)

        # Determine errors and errors
        mse = self.loss(y_val_est, y_val)
        mse = mse.data.numpy() #mean 
        print(mse)
        return mse

    def plot_learning_curve(self,ax,learning_curve,label,color):
        h, = ax.plot(learning_curve, color=color)
        h.set_label(label)
        ax.set_xlabel('Iterations')
        ax.set_xlim((0, self.n_batches))
        ax.set_ylabel('Loss')
        ax.set_title('Learning curves')


    
     

##################################################################################################################################

#Reference for Callback and EarlyStopping class:

#https://github.com/ncullen93/torchsample/blob/ea4d1b3975f68be0521941e733887ed667a1b46e/torchsample/callbacks.py#L348


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    
    
    
class EarlyStopping(Callback):
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        super(EarlyStopping, self).__init__()

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.trainer._stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                (self.stopped_epoch))


##################################################################################################################################



    