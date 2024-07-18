# https://github.com/UKPLab/pytorch-bertflow/blob/master/tflow_utils.py

import torch
from typing import Callable, Iterable, Tuple
from torch import nn
from torch.optim import Optimizer
from torch.nn import functional as F
from transformers import AutoModel
from torch.optim.lr_scheduler import LambdaLR
from math import log, pi, exp
import numpy as np
import random
from scipy import linalg as la
import os




def compute_unconditional_prior(z):
    h = z.new_zeros(z.shape)
    prior_dist = torch.distributions.normal.Normal(h, torch.exp(h))
    return prior_dist


def nll(sample):
    return 0.5*torch.sum(torch.pow(sample, 2), dim=[1,2,3])


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        
    
class VectorActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, **kwargs):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False, conditioning=None):
        if len(input.shape) == 2:
            input = input[:, :, None, None]
        if not reverse:
            if len(input.shape) == 4:
                # print(input.shape)
                _, _, height, width = input.shape
            else:
                # print(input.shape)
                input = input[:, None, None, None]
                _, _, height, width = input.shape
            if self.initialized.item() == 0:
                self.initialize(input)
                self.initialized.fill_(1)
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            if not self.logdet:
                return (self.scale * (input + self.loc)).squeeze()
            return (self.scale * (input + self.loc)).squeeze(), logdet
        else:
            return self.reverse(input)

    def reverse(self, output, conditioning=None):
        return (output / self.scale - self.loc).squeeze()



# A random permutation
class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]


class DoubleVectorCouplingBlock(nn.Module):
    """In contrast to VectorCouplingBlock, this module assures alternating chunking in upper and lower half."""
    def __init__(self, in_channels, hidden_dim=512, depth=2, use_hidden_bn=False, n_blocks=2):
        super(DoubleVectorCouplingBlock, self).__init__()
        assert in_channels % 2 == 0
        self.s = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=depth, hidden_dim=hidden_dim,
                                                       use_tanh=True) for _ in range(n_blocks)])
        self.t = nn.ModuleList([BasicFullyConnectedNet(dim=in_channels // 2, depth=depth, hidden_dim=hidden_dim,
                                                       use_tanh=False) for _ in range(n_blocks)])

    def forward(self, x, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                scale = self.s[i](x[idx_apply])
                x_ = x[idx_keep] * (scale.exp()) + self.t[i](x[idx_apply])
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale.view(x.size(0), -1), dim=1)
                logdet = logdet + logdet_
            return x, logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                x_ = (x[idx_keep] - self.t[i](x[idx_apply])) * (self.s[i](x[idx_apply]).neg().exp())
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x


class Flow(nn.Module):
    def __init__(self, module_list, in_channels, hidden_dim, hidden_depth):
        super(Flow, self).__init__()
        self.in_channels = in_channels
        self.flow = nn.ModuleList(
            [module(in_channels, hidden_dim=hidden_dim, depth=hidden_depth) for module in module_list])

    def forward(self, x, condition=None, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.flow)):
                x, logdet_ = self.flow[i](x)
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(len(self.flow))):
                x = self.flow[i](x, reverse=True)
            return x  


class EfficientVRNVP(nn.Module):
    def __init__(self, module_list, in_channels, n_flow, hidden_dim, hidden_depth):
        super().__init__()
        assert in_channels % 2 == 0
        self.flow = nn.ModuleList([Flow(module_list, in_channels, hidden_dim, hidden_depth) for n in range(n_flow)])

    def forward(self, x, condition=None, reverse=False):
        if not reverse:
            logdet = 0
            for i in range(len(self.flow)):
                x, logdet_ = self.flow[i](x, condition=condition)
                logdet = logdet + logdet_
            return x, logdet
        else:
            for i in reversed(range(len(self.flow))):
                x = self.flow[i](x, condition=condition, reverse=True)
            return x, None

    def reverse(self, x, condition=None):
        return self.flow(x, condition=condition, reverse=True)




class FactorLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rho = retrieve(config, "rho", default=0.975)

    def forward(self, samples, logdets, factors):
        sample1 = samples[0]
        logdet1 = logdets[0]
        nll_loss1 = torch.mean(nll(torch.cat(sample1, dim=1)))
        assert len(logdet1.shape) == 1
        nlogdet_loss1 = -torch.mean(logdet1)
        loss1 = nll_loss1 + nlogdet_loss1

        sample2 = samples[1]
        logdet2 = logdets[1]

        # nll_loss2 = torch.mean(nll(torch.cat(sample2, dim=1)))
        # assert len(logdet2.shape) == 1
        # nlogdet_loss2 = -torch.mean(logdet2)
        # loss2 = nll_loss2 + nlogdet_loss2
        # print(torch.tensor(((factors==0) | ((factors<0) & (factors!=-0)))))
        # print(factors)
        factor_mask = [
                torch.tensor([((factors==i) | ((factors<0) & (factors!=-i)))])[:,None,None,None].to(
                    sample2[i]) for i in range(len(sample2))]
        # for i in range(len(sample2)):
        #     print("i",i)
        #     print(sample2[i], "sample2[i]", sample2[i] - factor_mask[i]*self.rho*sample1[i], "sample1[i]", sample1[i])
        sample2_cond = [
                sample2[i] - factor_mask[i]*self.rho*sample1[i]
                for i in range(len(sample2))]
        nll_loss2 = [nll(sample2_cond[i]) for i in range(len(sample2_cond))]
        nll_loss2 = [nll_loss2[i]/(1.0-factor_mask[i][:,0,0,0]*self.rho**2)
                for i in range(len(sample2_cond))]
        nll_loss2 = [torch.mean(nll_loss2[i])
                for i in range(len(sample2_cond))]
        nll_loss2 = sum(nll_loss2)
        assert len(logdet2.shape) == 1
        nlogdet_loss2 = -torch.mean(logdet2)
        loss2 = nll_loss2 + nlogdet_loss2
        # print("loss1",loss1)
        # print("loss2",loss2)
        loss = (loss1 + loss2)

        log = {"images": {},
                "scalars": {
                    "loss": loss, "loss1": loss1, "loss2": loss2,
                }}
        def train_op():
            pass
        return loss, log



class VectorTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.config = config

        self.in_channel = retrieve(config, "in_channel")
        self.n_flow = retrieve(config, "n_flow")
        self.depth_submodules = retrieve(config, "hidden_depth")
        self.hidden_dim = retrieve(config, "hidden_dim")
        modules = [VectorActNorm, DoubleVectorCouplingBlock, Shuffle]
        self.realnvp = EfficientVRNVP(modules, self.in_channel, self.n_flow, self.hidden_dim,
                                   hidden_depth=self.depth_submodules)

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        input = input.squeeze()
        out, logdet = self.realnvp(input)
        return out[:, :, None, None], logdet

    def reverse(self, out):
        out = out.squeeze()
        return self.realnvp(out, reverse=True)[0][:, :, None, None]


class FactorTransformer(VectorTransformer):
    def __init__(self, config):
        super().__init__(config)
        self.n_factors = retrieve(config, "n_factors", default=2)
        # self.factor_dim = retrieve(config, "factor_dim")
        # self.factor_config = retrieve(config, "factor_config", default=list())

    def forward(self, input):
        out, logdet = super().forward(input)
        # if self.factor_config not None:
        # out = torch.split(out, self.factor_dim, dim=1)
        # # else:
        out = torch.chunk(out, self.n_factors, dim=1)
        return out, logdet

    def reverse(self, out):
        out = torch.cat(out, dim=1)
        return super().reverse(out)
    
        

# *************
class FactorTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor_loss = FactorLoss(config)
        self.glow = FactorTransformer(config)
    def forward(self, inputs, reverse=False, return_loss=True):
        if not reverse:
            if return_loss:
            # def step_op(self, *args, **kwargs):
            #     # get inputs
            #     inputs_factor = dict()
                self.factor = 0
                z_factor = dict()
                z_ss_factor = dict()
                logdet_factor = dict()
                factors = self.factor
                # for k in ["example1", "example2"]:
                for k in [0,1]: #"example1", "example2"
                    # print("first_input", inputs.shape)
                    n_inputs = inputs[:,k]
                    # print("input", n_inputs.shape)
                    z_ss, logdet = self.glow(n_inputs)
                    z_ss_factor[k] = z_ss
                    logdet_factor[k] = logdet
        
                # print(z_ss_factor.shape)
                # print(len(z_ss_factor), len(z_ss_factor[0]), len(z_ss_factor[0][0]), len(z_ss_factor[0][0][0]), len(z_ss_factor[0][0][0][0]))
                loss, log = self.factor_loss(z_ss_factor, logdet_factor, factors)
                # print(z_ss_factor)
                return z_ss_factor, loss
            else:
                self.factor = 0
                z_factor = dict()
                z_ss_factor = dict()
                logdet_factor = dict()
                factors = self.factor
                # for k in [0,1]: #"example1", "example2"
                # print("first_input", inputs.shape)
                n_inputs = inputs[:]
                # print("input", n_inputs.shape)
                z_ss, logdet = self.glow(n_inputs)
                z_ss_factor = z_ss
                logdet_factor = logdet
        
                # print(z_ss_factor.shape)
                # print(len(z_ss_factor), len(z_ss_factor[0]), len(z_ss_factor[0][0]), len(z_ss_factor[0][0][0]), len(z_ss_factor[0][0][0][0]))
                # loss, log = self.factor_loss(z_ss_factor, logdet_factor, factors)
                # print(1)
                # print(z_ss_factor)
                # return z_ss_factor, loss
                return z_ss
        else:
            z_ss = self.glow.reverse(inputs)
            return z_ss
        
# ************





class AdamWeightDecayOptimizer(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                m, v = state['m'], state['v']  # m, v
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                lr = group['lr']

                next_m = beta1 * m + (1 - beta1) * grad
                next_v = beta2 * v + (1 - beta2) * (grad ** 2)

                update = next_m / (torch.sqrt(next_v) + eps)
                if weight_decay != 0:
                    update += weight_decay * p

                updata_with_lr = lr * update
                p.add_(-updata_with_lr)
                state['m'] = next_m
                state['v'] = next_v
                state['step'] += 1
        return loss


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success