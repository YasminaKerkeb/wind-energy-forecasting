
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset



class PowerNNRegressor(nn.Module):
    
    """
    Base class for Neural Network model.
        
    """
    
    def __init__(self,model, loss_fn, n_epochs, max_iter,optimizer="Adam", tolerance=1e-6,learning_rate=0.001):
        """
        Initialize a PyTorch neural network model given a loss function, optimizer and other parameters for training
        
        Params
        ------
        
        model:          A function defining the model using torch.Sequential
        loss_fn:        string representing the supervised learning problem among ("regression", "binary" or "multiclass")
        optimizer:      string representing the Optimizer algorithm (Exp: "Adam")
        n_epochs:       An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned (default 3).
        max_iter:       An integer specifying the number of iterations in every epoch
                        to do 
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
        self.max_iter=max_iter
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
    

    def train(self,X_train,y_train):
        N,M=X_train.shape
        best_final_loss=1e100
        logging_frequency=self.max_iter/10
        # The dataloaders handle shuffling, batching, etc...
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
            for i in range(self.max_iter):
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
                best_net = self.state_dict()
                best_final_loss = loss_eval.item()
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


    
     

