import numpy as np

def numerical_gradient(model, X, Y, loss_func, eps=1e-5):
    grads = []
    for param in model.parameters():
        grad_W = np.zeros_like(param['W'])
        grad_b = np.zeros_like(param['b'])
        # W
        for i in range(param['W'].shape[0]):
            for j in range(param['W'].shape[1]):
                old = param['W'][i,j]
                param['W'][i,j] = old + eps
                loss_plus = loss_func(model.forward(X), Y)
                param['W'][i,j] = old - eps
                loss_minus = loss_func(model.forward(X), Y)
                grad_W[i,j] = (loss_plus - loss_minus) / (2*eps)
                param['W'][i,j] = old
        # b
        for i in range(param['b'].shape[1]):
            old = param['b'][0,i]
            param['b'][0,i] = old + eps
            loss_plus = loss_func(model.forward(X), Y)
            param['b'][0,i] = old - eps
            loss_minus = loss_func(model.forward(X), Y)
            grad_b[0,i] = (loss_plus - loss_minus) / (2*eps)
            param['b'][0,i] = old
        grads.append({'dW': grad_W, 'db': grad_b})
    return grads
