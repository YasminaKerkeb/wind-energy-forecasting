
import sklearn.metrics.cluster as cluster_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from matplotlib.pyplot import contourf
from matplotlib import cm
import torch


def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, n_epochs = 10000, tolerance=1e-6):
    
    """
    Train a neural network with PyTorch based on training set consisting of observations X and class y.
    Usage:
    We assume that the dataset has already been split to train and test data and transformed to PyTorch tensors
    Args: 
        model:          A function defining the model using torch.Sequential
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary 
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        X:              A train vector transformed to torch tensor
        y:              A target vector transformed to torch tensor
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        n_epochs:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerance:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
    """
    
    #Parameters for training
    best_final_loss=1e100
    epoch_frequency=1000
    
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r+1, n_replicates))
        # Initialize a model 
        net = model()
        
        #Use Xavier_uniform weight initialization
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        #We'll optimize the weights using Adam-algorithm, which is widely used
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        
 
        #Initialize the model
        for i in range(n_epochs):
            y_est = net(X) # forward propagation and predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # save loss

            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value

            # display loss with some frequency:
            if (i != 0) & ((i+1) % epoch_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
  

        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)

        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve

