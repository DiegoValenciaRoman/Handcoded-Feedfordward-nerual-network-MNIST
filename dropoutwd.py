from torch.utils.data import DataLoader
from random import randint
from matplotlib.pyplot import subplots
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import sys
import torch.nn as nn
import numpy as np
import math
from torch.nn import Parameter, ParameterList
import torch
import matplotlib.pyplot as plt
from timeit import default_timer as timer
plt.ion()


class SGD():
    def __init__(self, parameters, lr, beta=0):
        self.parameters = [p for p in parameters if p is not None]
        self.lr = lr
        self.beta = beta

    def step(self):
        for p in self.parameters:
            p.data = (1-self.beta)*p.data - self.lr * p.grad


def sig(T, gradient=False):
    if gradient:
        sigT = sig(T)
        return sigT * (1 - sigT)
    return torch.reciprocal(1 + torch.exp(-1 * T))


def tanh(T, gradient=False):
    if gradient:
        tanhT = tanh(T)
        return 1 - tanhT * tanhT
    E = torch.exp(T)
    e = torch.exp(-1 * T)
    return (E - e) * torch.reciprocal(E + e)


def relu(T, grad=False):
    if grad:
        outT = torch.zeros_like(T).to("cpu")
        outT[T >= 0] = 1
        return outT
    return torch.max(T, torch.zeros_like(T))


def swish(T, beta=1, gradient=False):
    if gradient:
        sigbT = sig(beta * T)
        swishT = T * sigbT
        return sigbT + beta * swishT * (1 - sigbT), swishT * (T - swishT)
    return T * torch.reciprocal(1 + torch.exp(-beta * T))


def celu(T, alpha=1, gradient=False):
    if alpha == 0:
        raise ValueError("alpha cannot be 0")

    zeros = torch.zeros_like(T)
    Talpha = T / alpha

    if gradient:
        e = Talpha.exp()
        d_dx = torch.ones_like(T)
        d_dx[T < 0] = e[T < 0]
        zeros[T < 0] = (celu(T)[T < 0] - T[T < 0] * e[T < 0]) / alpha
        return d_dx, zeros  # d_dx, d_da

    return torch.max(zeros, T) + torch.min(zeros, alpha * (Talpha).expm1())


def softmax(T, dim, estable=True):
    if estable:
        # keepdim=True => output has dim with size 1. Otherwise, dim is squeezed
        T -= T.max(dim=dim, keepdim=True)[0]
    exp = torch.exp(T)
    # keepdim=True => output has dim with size 1. Otherwise, dim is squeezed
    return exp / torch.sum(exp, dim=dim, keepdim=True)


class FFNN(nn.Module):
    def __init__(self, F, l_h, l_a, C, l_a_params=None, keep_prob=None):
        # debes crear los parámetros necesarios para hacer
        # dropout en cada capa dependiendo de keep_prob
        super(FFNN, self).__init__()
        sizes = [F] + l_h + [C]
        self.Ws = torch.nn.ParameterList([torch.nn.Parameter(
            torch.randn(sizes[i], sizes[i+1])) for i in range(len(sizes)-1)])
        self.bs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(h)) for h in sizes[1:]])
        self.fs = l_a
        self.kp = keep_prob
        print("kp", self.kp)
        if l_a_params is not None:
            self.fs_ps_mask = [Parameter(torch.tensor(
                p)) if p else None for p in l_a_params]
        else:
            self.fs_ps_mask = [None for _ in l_a]
        self.fs_ps = ParameterList([p for p in self.fs_ps_mask if p])

    def load_weights(self, Ws, U, bs, c):
        self.Ws = torch.nn.ParameterList(
            [torch.nn.Parameter(W) for W in Ws + [U]])
        self.bs = torch.nn.ParameterList(
            [torch.nn.Parameter(b) for b in bs + [c]])

    def forward(self, x, predict=False, output_layer=0):
        # debes modificar esta función para considerar el dropout
        # y preocuparte de no considerarlo cuando (predict=True)
        self.cache = []
        self.cacheU = []
        for i, (W, b, f) in enumerate(zip(self.Ws[:-1], self.bs[:-1], self.fs)):
            self.cacheU.append(torch.mm(x, W) + b)
            self.cache.append(torch.mm(x, W) + b)
            # agregando inverted dropout
            if(predict):
                x = f(torch.mm(x, W) + b)
            else:
                x = f(torch.mm(x, W) + b)
                dpvector = torch.rand(x.shape[1]) < self.kp[i]
                x = x*dpvector.to("cpu")
                x = x/self.kp[i]
        return softmax(torch.mm(x, self.Ws[-1]) + self.bs[-1], dim=1)

    def backward(self, x, y, y_pred):
        grads = []
        params = []
        b = x.size()[0]
        dL_du3 = (1/b) * (y_pred - y)
        dL_dU = self.fs[len(self.fs) -
                        1](self.cache[len(self.cache)-1]).t() @ dL_du3
        dL_dc = torch.sum(dL_du3, 0)
        grads.insert(0, dL_dc)
        grads.insert(0, dL_dU)

        dL_ant = dL_du3
        for i in range(len(self.cache)-1, 0, -1):
            dL_dhk = dL_ant @ self.Ws[i+1].t()
            dL_duk = dL_dhk * self.fs[i](self.cache[i], grad=True)
            dL_ant = dL_duk
            dL_dWk = self.fs[i](self.cache[i-1]).t() @ dL_duk
            dL_dbk = torch.sum(dL_duk, 0)
            grads.insert(0, dL_dbk)
            grads.insert(0, dL_dWk)

        dL_dh1 = dL_ant @ self.Ws[1].t()
        dL_du1 = dL_dh1 * self.fs[0](self.cache[0], grad=True)
        dL_dW1 = x.t() @ dL_du1
        dL_db1 = torch.sum(dL_du1, 0)
        grads.insert(0, dL_db1)
        grads.insert(0, dL_dW1)

        for a, b in zip(self.Ws, self.bs):
            params.append(a)
            params.append(b)
        for p, g in zip(params, grads):
            p.grad = g

##old approach, tambien funciona
"""
    def backward(self, x, y, y_pred):
        current_grad = (y_pred - y) / y.size(0)

        for i in range(len(self.Ws)-1, 0, -1):
            if self.fs_ps_mask[i-1] is None:
                self.Ws[i].grad = self.fs[i -
                                          1](self.cacheU[i-1]).t() @ current_grad
            else:
                self.Ws[i].grad = self.fs[i-1](self.cacheU[i-1],
                                               self.fs_ps_mask[i-1].item()).t()  @ current_grad
            self.bs[i].grad = current_grad.sum(dim=0)
            h_grad = current_grad @ self.Ws[i].t()

            if self.fs_ps_mask[i-1] is None:
                current_grad = self.fs[i -
                                       1](self.cacheU[i-1], gradient=True) * h_grad
            else:
                current_grad, p_grad = self.fs[i-1](self.cacheU[i-1],
                                                    self.fs_ps_mask[i-1], gradient=True)
                current_grad *= h_grad
                self.fs_ps_mask[i-1].grad = (p_grad * h_grad).sum()

        self.Ws[0].grad = x.t() @ current_grad
        self.bs[0].grad = current_grad.sum(dim=0)"""


def plot_results(loss, acc):
    f1 = plt.figure(1)
    ax1 = f1.add_subplot(111)
    ax1.set_title("Loss")
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(loss, c='r')
    f1.show()

    f2 = plt.figure(2)
    ax2 = f2.add_subplot(111)
    ax2.set_title("Accuracy")
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('acc')
    ax2.plot(acc, c='b')
    f2.show()
# Aqui el codigo para entrenar en MNIST


def CELoss(Q, P, estable=True, epsilon=1e-8):
    N = Q.shape[0]
    if estable:
        Q = Q.clamp(epsilon, 1-epsilon)
    return -(P * Q.log()).sum()/N


# Importamos funcionalidades útiles para mirar los datos.
# Descarga y almacena el conjunto de prueba de MNIST.
dataset = MNIST('mnist', train=True, transform=ToTensor(), download=True)
print('Cantidad total de datos:', len(dataset))

# data_loader = DataLoader(dataset, batch_size=1000)


def entrenar_FFNN(red, dataset, optimizador, epochs=1, batch_size=1, reports_every=1, device='cpu'):
    red.to(device)
    data = DataLoader(dataset, batch_size, shuffle=True)
    total = len(dataset)
    tiempo_epochs = 0
    loss, acc = [], []
    for e in range(1, epochs+1):
        inicio_epoch = timer()

        for x, y in data:
            x, y = x.view(x.size(0), -1).float().to(device), y.to(device)

            y_pred = red(x)

            y_onehot = torch.zeros_like(y_pred)
            y_onehot[torch.arange(x.size(0)), y] = 1.

            red.backward(x, y_onehot, y_pred)

            optimizador.step()

        tiempo_epochs += timer() - inicio_epoch

        if e % reports_every == 0:
            X = dataset.data.view(len(dataset), -1).float().to(device)
            Y = dataset.targets.to(device)

            Y_PRED = red.forward(X).to(device)

            Y_onehot = torch.zeros_like(Y_PRED)
            Y_onehot[torch.arange(X.size(0)), Y] = 1.

            L_total = CELoss(Y_PRED, Y_onehot)
            loss.append(L_total)
            diff = Y-torch.argmax(Y_PRED, 1)
            errores = torch.nonzero(diff).size(0)

            Acc = 100*(total-errores)/total
            acc.append(Acc)

            sys.stdout.write(
                '\rEpoch:{0:03d}'.format(e) + ' Acc:{0:.2f}%'.format(Acc)
                + ' Loss:{0:.4f}'.format(L_total)
                  + ' Tiempo/epoch:{0:.3f}s'.format(tiempo_epochs/e))

    return loss, acc



mnist_model = FFNN(784, [600, 200], [relu, relu],
                   10, keep_prob=[1.0, 0.5, 0.5])

#probar parametros
mnist_optimizer = SGD(mnist_model.parameters(), lr=1e-3, beta=0.9)
with torch.no_grad():
    mnist_loss, mnist_acc = entrenar_FFNN(
        mnist_model, dataset, mnist_optimizer, epochs=30, batch_size=1000)
#print("\n\t\t\tModelo sin dropout ni WD")
plot_results(mnist_loss, mnist_acc)
