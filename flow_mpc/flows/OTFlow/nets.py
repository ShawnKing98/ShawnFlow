# neural network to model the potential function
import torch
import torch.nn as nn
import copy
import math


def antiderivTanh(x):  # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))


def derivTanh(x):  # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow(torch.tanh(x), 2)


class ResNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, num_layers=2):
        super().__init__()

        if num_layers < 2:
            print("num_layers must be an integer >= 2")
            exit(1)

        self.d = input_dim
        self.m = hidden_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(input_dim + 1 + context_dim, hidden_dim, bias=True))  # opening layer
        self.layers.append(nn.Linear(hidden_dim + context_dim, hidden_dim, bias=True))  # resnet layers
        for i in range(num_layers - 2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.num_layers - 1)  # step size for the ResNet

    def forward(self, x, context):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """
        h = torch.cat((x, context), dim=1)
        x = self.act(self.layers[0].forward(h))

        for i in range(1, self.num_layers):
            h = torch.cat((x, context), dim=1)
            x = x + self.h * self.act(self.layers[i](h))

        return x


class Phi(nn.Module):
    def __init__(self, input_dim, hidden_dim, context_dim, num_layers, r=10):
        """
            neural network approximating Phi (see Eq. (9) in our paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        """
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        r = min(r, input_dim + 1)  # if number of dimensions is smaller than default r, use that

        self.A = nn.Parameter(torch.zeros(r, input_dim + 1), requires_grad=True)
        self.A = nn.init.xavier_uniform_(self.A)
        self.c = nn.Linear(input_dim + 1, 1, bias=True)  # b'*[x;t] + c
        self.w = nn.Linear(hidden_dim, 1, bias=False)

        self.net = ResNN(input_dim, hidden_dim, context_dim, num_layers=num_layers)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data = torch.zeros(self.c.bias.data.shape)

    def forward(self, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """

        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A)  # A'A

        return self.w(self.N(x)) + 0.5 * torch.sum(torch.matmul(x, symA) * x, dim=1, keepdims=True) + self.c(x)

    def trHess(self, x, context, justGrad=False):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh
        m = self.net.layers[0].weight.shape[0]
        nex = x.shape[0]  # number of examples in the batch
        d = x.shape[1] - 1
        symA = torch.matmul(self.A.t(), self.A)

        u = []  # hold the u_0,u_1,...,u_M for the forward pass
        z = self.net.num_layers * [None]  # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        h = torch.cat((x, context), dim=1)
        opening = self.net.layers[0].forward(h)  # K_0 * S + b_0
        u.append(self.net.act(opening))  # u0
        feat = u[0]

        for i in range(1, self.net.num_layers):
            h = torch.cat((feat, context), dim=1)
            feat = feat + self.net.h * self.net.act(self.net.layers[i](h))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening)  # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(self.net.num_layers - 1, 0, -1):  # work backwards, placing z_i in appropriate spot
            if i == self.net.num_layers - 1:
                term = self.w.weight.t()
            else:
                term = z[i + 1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            h = torch.cat((u[i-1], context), dim=1)
            # layers are 64 x 320
            # only want 64 x 64 component to get grads

            selector = torch.cat((torch.eye(self.hidden_dim), torch.zeros(self.hidden_dim, self.context_dim)), dim=1).to(device=x.device)
            z[i] = term + self.net.h * torch.mm(selector, torch.mm(self.net.layers[i].weight.t(),
                                                torch.tanh(self.net.layers[i].forward(h)).t() * term))

        # z_0 = K_0' diag(...) z_1

        z[0] = torch.mm(self.net.layers[0].weight.t(), tanhopen.t() * z[1])
        selector = torch.cat((torch.eye(x.shape[1]), torch.zeros(x.shape[1], self.context_dim)), dim=1).to(
            device=x.device)
        z[0] = torch.mm(selector, z[0])
        grad = z[0] + torch.mm(symA, x.t()) + self.c.weight.t()

        if justGrad:
            return grad.t()

        # -----------------
        # trace of Hessian
        # -----------------

        # t_0, the trace of the opening layer
        Kopen = self.net.layers[0].weight[:, 0:d]  # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp = derivTanh(opening.t()) * z[1]
        trH = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1))  # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()  # act'( K_0 * S + b_0 )
        Jac = Kopen.unsqueeze(2) * temp.unsqueeze(1)  # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1, self.net.num_layers):
            Wi = self.net.layers[i].weight[:, :m]
            KJ = torch.mm(Wi, Jac.reshape(m, -1))
            KJ = KJ.reshape(m, -1, nex)
            if i == self.net.num_layers - 1:
                term = self.w.weight.t()
            else:
                term = z[i + 1]

            h = torch.cat((u[i-1], context), dim=1)
            temp = self.net.layers[i].forward(h).t()  # (K_i * u_{i-1} + b_i)
            t_i = torch.sum((derivTanh(temp) * term).reshape(m, -1, nex) * torch.pow(KJ, 2), dim=(0, 1))
            trH = trH + self.net.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + self.net.h * torch.tanh(temp).reshape(m, -1, nex) * KJ  # update Jacobian

        return grad.t(), trH + torch.trace(symA[0:d, 0:d])
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )


def print_grad(var):
    if var.requires_grad:
        var.register_hook(lambda x: print(x))
