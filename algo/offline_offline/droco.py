import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributions.transforms import Transform

from typing import Dict, List, Tuple, Union, Optional, Type
from functools import partial
from tqdm import trange
import itertools
import os


def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which axis is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class LinearEnsemble(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 3,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        An Ensemble linear layer.
        For inputs of shape (B, H) will return (E, B, H) where E is the ensemble size
        See https://github.com/pytorch/pytorch/issues/54147
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((ensemble_size, 1, out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The default torch init for Linear is a complete mess
        # https://github.com/pytorch/pytorch/issues/57109
        # If we use the same init, we will end up scaling incorrectly
        # 1. Compute the fan in of the 2D tensor = dim 1 of 2D matrix (0 index)
        # 2. Comptue the gain with param=math.sqrt(5.0)
        #   This returns math.sqrt(2.0 / 6.0) = sqrt(1/3)
        # 3. Compute std = gain / math.sqrt(fan) = sqrt(1/3) / sqrt(in).
        # 4. Compute bound as math.sqrt(3.0) * std = 1 / in di
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        elif len(input.shape) > 3:
            raise ValueError("LinearEnsemble layer does not support inputs with more than 3 dimensions.")
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "ensemble_size={}, in_features={}, out_features={}, bias={}".format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )


class LayerNormEnsemble(nn.Module):
    """
    This is a re-implementation of the Pytorch nn.LayerNorm module with suport for the Ensemble dim.
    We need this custom class since we need to normalize over normalize dims, but have multiple weight/bais
    parameters for the ensemble.

    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: int,
        ensemble_size: int = 3,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert isinstance(normalized_shape, int), "Currently EnsembleLayerNorm only supports final dim int shapes."
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.ensemble_size = ensemble_size
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty((self.ensemble_size, 1) + self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty((self.ensemble_size, 1) + self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.repeat(self.ensemble_size, 1, 1)
        elif len(x.shape) > 3:
            raise ValueError("LayerNormEnsemble layer does not support inputs with more than 3 dimensions.")
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)  # (E, B, *normalized shape)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ensemble_size: int = 3,
        hidden_layers: List[int] = [256, 256],
        act: nn.Module = nn.ReLU,
        dropout: float = 0.0,
        normalization: Optional[Type[nn.Module]] = None,
        output_act: Optional[Type[nn.Module]] = None,
    ):
        """
        An ensemble MLP
        Returns values of shape (E, B, H) from input (B, H)
        """
        super().__init__()
        # Change the normalization type to work over ensembles
        assert normalization is None or normalization is LayerNormEnsemble, "Ensemble only support EnsembleLayerNorm"
        net = []
        last_dim = input_dim
        for dim in hidden_layers:
            net.append(LinearEnsemble(last_dim, dim, ensemble_size=ensemble_size))
            if dropout > 0.0:
                net.append(nn.Dropout(dropout))
            if normalization is not None:
                net.append(normalization(dim, ensemble_size=ensemble_size))
            net.append(act())
            last_dim = dim
        net.append(LinearEnsemble(last_dim, output_dim, ensemble_size=ensemble_size))
        if output_act is not None:
            net.append(output_act())
        self.net = nn.Sequential(*net)
        self._has_output_act = False if output_act is None else True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def last_layer(self) -> torch.Tensor:
        if self._has_output_act:
            return self.net[-2]
        else:
            return self.net[-1]



class ContrastiveInfo(nn.Module):
    def __init__(
        self, 
        state_dim:          int,
        action_dim:         int,
        repr_dim:           int,
        ensemble_size:      int = 2,
        repr_norm:          bool = False,
        repr_norm_temp:     bool = True,
        ortho_init:         bool = False,
        output_gain:        Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.state_dim          = state_dim
        self.action_dim         = action_dim        
        self.repr_dim           = repr_dim
        self.ensemble_size      = ensemble_size
        self.repr_norm          = repr_norm
        self.repr_norm_temp     = repr_norm_temp
        
        input_dim_for_sa        = self.state_dim + self.action_dim
        input_dim_for_ss        = self.state_dim

        if self.ensemble_size > 1:
            self.encoder_sa     = EnsembleMLP(input_dim_for_sa, repr_dim, ensemble_size=ensemble_size, **kwargs)
            self.encoder_ss     = EnsembleMLP(input_dim_for_ss, repr_dim, ensemble_size=ensemble_size, **kwargs)
        else:
            self.encoder_sa     = MLPNetwork(input_dim_for_sa, repr_dim, **kwargs)
            self.encoder_ss     = MLPNetwork(input_dim_for_ss, repr_dim, **kwargs)

        self.ortho_init    = ortho_init
        self.output_gain   = output_gain
        self.register_parameter()

    def register_parameter(self) -> None:
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))
    
    def encode(self, obs: torch.Tensor, action: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
        sa_repr      = self.encoder_sa(torch.cat([obs, action], dim=-1))
        ss_repr      = self.encoder_ss(ss)
        if self.repr_norm:
            sa_repr     =   sa_repr / torch.linalg.norm(sa_repr, dim=-1, keepdim=True)
            ss_repr     =   ss_repr / torch.linalg.norm(ss_repr, dim=-1, keepdim=True)
            if self.repr_norm_temp:
                raise NotImplementedError("The Running normalization is not implemented")
        return sa_repr, ss_repr

    def combine_repr(self, sa_repr: torch.Tensor, ss_repr: torch.Tensor) -> torch.Tensor:
        if len(sa_repr.shape) ==2 and len(ss_repr.shape) ==2:
            return torch.einsum('iz,jz->ij', sa_repr, ss_repr)
        else:
            return torch.einsum('eiz,ejz->eij', sa_repr, ss_repr)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, ss: torch.Tensor, return_repr: bool = False) -> torch.Tensor:
        sa_repr, ss_repr = self.encode(obs, action, ss)    #   [E, B1, Z], [E, B2, Z]
        if return_repr:
            return self.combine_repr(sa_repr, ss_repr), sa_repr, ss_repr
        else:
            return self.combine_repr(sa_repr, ss_repr)           #   [E, B1, B2]


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        
        return action * self.max_action, logprob, mean * self.max_action
    
    def bc_loss(self, state, action):
        mu_logstd = self.network(state)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        pred_action = torch.tanh(mu)

        return (pred_action - action)**2

class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class ValueFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


# dynamics model相关的模块
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Module(nn.Module):
    def save(self, f: str, prefix: str = '', keep_vars: bool = False) -> None:
        state_dict = self.state_dict(prefix= prefix, keep_vars=keep_vars)
        torch.save(state_dict, f)

    def load(self, f: str, map_location, strict: bool = True) -> None:
        state_dict = torch.load(f, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        ensemble_size: int, 
        weight_decay: float = 0., 
        bias: bool = True
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)


class Normalizer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim        = dim
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.zeros(dim))

    def fit(self, X: torch.tensor) -> None:
        assert len(X.shape)      == 2
        assert X.shape[1]   == self.dim
        device  =   self.mean.device
        self.mean.data.copy_(
            torch.mean(X, axis=0, keepdims=False)
        )
        self.std.data.copy_(
            torch.std(X, axis=0, keepdims=False)
        )
        self.std[self.std < 1e-12]  =   1.0

    def transform(self, x: Union[np.array, torch.tensor]) -> Union[np.array, torch.tensor]:
        if isinstance(x, np.ndarray):
            device  =   self.mean.device
            x       =   torch.from_numpy(x).float().to(device)
            return ((x - self.mean) / self.std).cpu().numpy()
        elif isinstance(x, torch.Tensor):
            return ((x - self.mean) / self.std)


def init_weights(m):
    def truncated_normal_init(
        t:  nn.Module, 
        mean:   float = 0.0, 
        std:    float = 0.01
    ):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class EnsembleModel(nn.Module):
    def __init__(
        self, 
        state_size:     int, 
        action_size:    int, 
        reward_size:    int, 
        ensemble_size:  int, 
        hidden_size:    int   = 200, 
        learning_rate:  float = 1e-3, 
        use_decay:      bool  = False,
        device:         str   = 'cuda'
    ):
        super(EnsembleModel, self).__init__()
        self.device      = device
        self.hidden_size = hidden_size
        self.output_dim  = state_size + reward_size
        # trunk layers
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.000025)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00005)
        self.nn3 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.nn4 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.000075)
        self.use_decay = use_decay
        # Add variance output
        self.nn5 = EnsembleFC(hidden_size, self.output_dim * 2, ensemble_size, weight_decay=0.0001)
        # min / max log var bounds
        self.max_logvar = nn.Parameter((torch.ones((1, self.output_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.output_dim)).float() * 10).to(device), requires_grad=False)
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # weight init
        self.apply(init_weights)
        self.swish = Swish()

    def forward(
        self, 
        x:           torch.tensor, 
        ret_log_var: bool = False
    ):
        nn1_output = self.swish(self.nn1(x))
        nn2_output = self.swish(self.nn2(nn1_output))
        nn3_output = self.swish(self.nn3(nn2_output))
        nn4_output = self.swish(self.nn4(nn3_output))
        nn5_output = self.nn5(nn4_output)

        mean    = nn5_output[:, :, :self.output_dim]
        logvar  = nn5_output[:, :, self.output_dim:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

    def loss(
        self, 
        mean:           torch.tensor, 
        logvar:         torch.tensor, 
        labels:         torch.tensor, 
        inc_var_loss:   bool = True
    ):
        """
            mean, logvar: [ensemble_size,  batch_size, |S| + |A|]
            labels:       [ensemble_size,  batch_size, |S| + 1]
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        self.optimizer.step()


class EnsembleDynamicsModel(Module):
    def __init__(
        self, 
        network_size:   int, 
        elite_size:     int, 
        state_size:     int, 
        action_size:    int, 
        reward_size:    int = 1, 
        hidden_size:    int = 200, 
        use_decay:      bool= False,
        device:         str = 'cuda',
    ):
        super(EnsembleDynamicsModel, self).__init__()
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.network_size = network_size
        self.elite_model_idxes = []
        self.ensemble_model = EnsembleModel(state_size, action_size, reward_size, network_size, hidden_size, use_decay=use_decay, device=device)
        self.scaler = Normalizer(dim=state_size + action_size)
        self.device = device

    def train(
        self, 
        inputs:                     torch.tensor, 
        labels:                     torch.tensor, 
        batch_size:                 int     = 256, 
        holdout_ratio:              float   = 0., 
        max_epochs_since_update:    int     = 5
    ):
        self._max_epochs_since_update   = max_epochs_since_update
        self._epochs_since_update       = 0
        self._state                     = {}
        self._snapshots                 = {i: (None, 1e10) for i in range(self.network_size)}

        num_holdout     = int(inputs.shape[0] * holdout_ratio)
        permutation     = torch.randperm(inputs.shape[0])
        inputs, labels  = inputs[permutation], labels[permutation]

        train_inputs, train_labels      = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels  = inputs[:num_holdout], labels[:num_holdout]

        self.scaler.fit(train_inputs)
        train_inputs    = self.scaler.transform(train_inputs)
        holdout_inputs  = self.scaler.transform(holdout_inputs)

        holdout_inputs = holdout_inputs[None, :, :].repeat([self.network_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.network_size, 1, 1])
        # for log
        all_holdout_losses  =   []


        train_idx = torch.vstack([torch.randperm(train_inputs.shape[0]) for _ in range(self.network_size)])
        
        losses = []
        for start_pos in range(0, train_inputs.shape[0], batch_size):
            idx = train_idx[:, start_pos: min(start_pos + batch_size, train_inputs.shape[0])]
            train_input = torch.from_numpy(train_inputs.cpu().numpy()[idx]).float().to(self.device)
            train_label = torch.from_numpy(train_labels.cpu().numpy()[idx]).float().to(self.device)
            mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
            loss, mse_loss = self.ensemble_model.loss(mean, logvar, train_label)
            losses.append(torch.mean(mse_loss).item())
            self.ensemble_model.train(loss)
        
        avg_loss = sum(losses) / len(losses)
        
        print(f"Loss: {avg_loss}")
                    
        self._track_head_loss(all_holdout_losses)

    def _save_best(
        self, 
        epoch:          int, 
        holdout_losses: torch.tensor    # [ensemble_size]
    ):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def _track_head_loss(
        self,
        holdout_losses:  List    # [<= max_train_epoch]
    )   -> None:
        self._current_mean_ensemble_losses = np.mean(holdout_losses)

    def predict(
        self,
        inputs:                 torch.Tensor, 
        batch_size:             float   = 1024, 
        factor_ensemble:        bool    = True
    ):
        if inputs.ndim == 2:
            B       = inputs.shape[0]
            inputs  = self.scaler.transform(inputs)
            ensemble_mean, ensemble_var = [], []
            for i in range(0, B, batch_size):
                input = inputs[i:min(i + batch_size, B)]
                b_mean, b_var = self.ensemble_model(
                    input[None, :, :].repeat([self.network_size, 1, 1]), 
                    ret_log_var=False
                )
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            ensemble_mean   = torch.cat(ensemble_mean, dim=1)    # concat along the batch_size axis
            ensemble_var    = torch.cat(ensemble_var, dim=1)

            if factor_ensemble:
                return ensemble_mean, ensemble_var              # [ensemble_size, batch_size, |S|+1]
            else:
                mean    = torch.mean(ensemble_mean, dim=0)
                var     = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
                return mean, var
        elif inputs.ndim == 3:
            assert inputs.shape[0] == self.network_size
            B       = inputs.shape[1]
            inputs  = self.scaler.transform(inputs)
            ensemble_mean, ensemble_var = [], []
            for i in range(0, B, batch_size):
                input = inputs[:, i:min(i + batch_size, B), :]
                b_mean, b_var = self.ensemble_model(
                    input,
                    ret_log_var=False
                )
                ensemble_mean.append(b_mean)
                ensemble_var.append(b_var)
            ensemble_mean   = torch.cat(ensemble_mean, dim=1)    # concat along the batch_size axis
            ensemble_var    = torch.cat(ensemble_var, dim=1)

            if factor_ensemble:
                return ensemble_mean, ensemble_var              # [ensemble_size, batch_size, |S|+1]
            else:
                mean    = torch.mean(ensemble_mean, dim=0)
                var     = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
                return mean, var
        else:
            raise ValueError            


def soft_update(src_model: nn.Module, tar_model: nn.Module, tau: float) -> None:
    for param_src, param_tar in zip(src_model.parameters(), tar_model.parameters()):
        param_tar.data.copy_(tau * param_src.data + (1 - tau) * param_tar.data)

def huber_loss(y_true, y_pred, delta=30.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    
    loss = torch.where(abs_error <= delta, 
                    0.5 * (error ** 2), 
                    delta * (abs_error - 0.5 * delta))
    
    return torch.mean(loss)


class DROCO(object):

    def __init__(self,
                 config,
                 device,
                 target_entropy=None,
                 ):
        self.config=  config
        self.device = device
        self.discount = config['gamma']
        self.tau = config['tau']
        self.target_entropy = target_entropy if target_entropy else -config['action_dim']
        self.update_interval = config['update_interval']

        # IQL hyperparameter
        self.lam = config['lam']
        self.temp = config['temp']
        
        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka value
        self.v_func = ValueFunc(config['state_dim'], config['action_dim'], hidden_size=config['hidden_sizes']).to(self.device)

        # aka actor
        self.policy = Policy(config['state_dim'], config['action_dim'], config['max_action'], hidden_size=config['hidden_sizes']).to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=config['critic_lr'])
        self.v_optimizer = torch.optim.Adam(self.v_func.parameters(), lr=config['critic_lr'])
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'])

        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, config['max_step'])
        
        
        # dynamics model
        self.model_config = config['model_config']
        self.dynamics_batch_size                    =       config['dynamics_batch_size']
        self.dynamics_holdout_ratio                 =       config['dynamics_holdout_ratio']
        self.dynamics_max_epochs_since_update       =       config['dynamics_max_epochs_since_update']
        self.max_epochs_since_update_decay_interval =       config['max_epochs_since_update_decay_interval']
        
        self.dynamics   =   EnsembleDynamicsModel( 
            network_size=   self.model_config['dynamics_ensemble_size'],
            elite_size  =   self.model_config['dynamics_elite_size'],
            state_size  =   config['state_dim'],
            action_size =   config['action_dim'],
            reward_size =   1,
            hidden_size =   self.model_config['dynamics_hidden_size'],
            use_decay   =   True,
            device      =   self.device
        ).to(self.device)
        
        

    def train_model(self, tar_replay_buffer,  current_step: int,) -> None:
        # decay the max epochs since update coefficient
        current_dynamics_max_epochs_since_update    =   max(
            0,
            self.dynamics_max_epochs_since_update - int(current_step / self.max_epochs_since_update_decay_interval)
        )

        s_batch, a_batch, next_s_batch, r_batch, not_done_batch = tar_replay_buffer.sample(tar_replay_buffer.size)
        delta_s_batch = next_s_batch - s_batch
        inputs      = torch.cat((s_batch, a_batch), dim=-1)
        labels      = torch.cat((r_batch, delta_s_batch), dim=-1)
        self.dynamics.train(
            inputs                  =   inputs,
            labels                  =   labels,
            batch_size              =   self.dynamics_batch_size,
            holdout_ratio           =   self.dynamics_holdout_ratio,
            max_epochs_since_update =   current_dynamics_max_epochs_since_update
        )
    
    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(self.device))
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_v_function(self, state_batch, action_batch, writer=None):
        with torch.no_grad():
            q_t1, q_t2 = self.target_q_funcs(state_batch, action_batch)
            q_t = torch.min(q_t1, q_t2)
            
        v = self.v_func(state_batch)
        adv = q_t - v
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar('train/adv', adv.mean(), self.total_it)
            writer.add_scalar('train/value', v.mean(), self.total_it)
        v_loss = asymmetric_l2_loss(adv, self.lam)
        return v_loss, adv

    def update_q_functions(self, src_state, src_action, src_reward, src_next_state, src_not_done, tar_state, tar_action, tar_reward, tar_next_state, tar_not_done, penalty, writer=None):
        with torch.no_grad():
            v_t = self.v_func(src_next_state)
            value_target = src_reward + src_not_done * self.discount * (v_t - penalty)
            
        q_1, q_2 = self.q_funcs(src_state, src_action)
        src_loss = (huber_loss(value_target, q_1, delta=self.config["huber_delta"])).mean() + (huber_loss(value_target, q_2, delta=self.config["huber_delta"])).mean()
        

        with torch.no_grad():
            v_t = self.v_func(tar_next_state)
            value_target = tar_reward + tar_not_done * self.discount * v_t
            
        q_1, q_2 = self.q_funcs(tar_state, tar_action)
        tar_loss = (0.5 * (value_target - q_1)**2).mean() + (0.5 * (value_target - q_2)**2).mean()
        
        return src_loss + tar_loss

    def update_policy(self, advantage_batch, state_batch, action_batch):
        exp_adv = torch.exp(self.temp * advantage_batch.detach()).clamp(max=100.0)
        bc_loss = self.policy.bc_loss(state_batch, action_batch)
        policy_loss = torch.mean(exp_adv * bc_loss)
        return policy_loss
    
    def log_q(self, src_replay_buffer, batch_size=128):
        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        q_1, q_2 = self.q_funcs(src_state, src_action)
        return q_1.mean()

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):
        
        self.total_it += 1

        src_state, src_action, src_next_state, src_reward, src_not_done = src_replay_buffer.sample(batch_size)
        tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = tar_replay_buffer.sample(batch_size)
        
        dyna_pred_mean, dyna_pred_var = self.dynamics.predict(inputs=torch.cat([src_state, src_action], dim=-1), factor_ensemble=True)   # [ensemble_size, batch_size, 1 + |S|]
        dyna_pred_samples               =   dyna_pred_mean + torch.ones_like(dyna_pred_var, device=self.device) * dyna_pred_var
        _, dyna_pred_delta_s  =   dyna_pred_samples[:, :, :1], dyna_pred_samples[:, :, 1:]
        dyna_pred_next_s                =   src_state + dyna_pred_delta_s # [E, B, S]
        dyna_pred_inf_next_s, min_indices = torch.min(dyna_pred_next_s, dim=0, keepdim=False) # [B,S]

        penalty = torch.zeros((2 * batch_size, 1)).to(self.device)
        penalty[:batch_size] = torch.clamp(self.config["penalty_coefficient"] * (self.v_func(src_next_state) - self.v_func(dyna_pred_inf_next_s)), min=0.0)


        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)
        reward = torch.cat([src_reward, tar_reward], 0)
        not_done = torch.cat([src_not_done, tar_not_done], 0)

        v_loss_step, adv = self.update_v_function(state, action, writer)
        self.v_optimizer.zero_grad()
        v_loss_step.backward()
        self.v_optimizer.step()

        q_loss_step = self.update_q_functions(src_state, src_action, src_reward, src_next_state, src_not_done, tar_state, tar_action, tar_reward, tar_next_state, tar_not_done, penalty, writer)

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss_step = self.update_policy(adv, state, action)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.v_func.state_dict(), filename + "_value")
        torch.save(self.v_optimizer.state_dict(), filename + "_value_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.policy_lr_schedule.state_dict(), filename + "_actor_lr_scheduler")

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.v_func.load_state_dict(torch.load(filename + "_value"))
        self.v_optimizer.load_state_dict(torch.load(filename + "_value_optimizer"))
        self.dynamics.load_state_dict(torch.load(filename + "_dynamics"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.policy_lr_schedule.load_state_dict(torch.load(filename + "_actor_lr_scheduler"))
        
    def save_dynamics(self, filename):
        torch.save(self.dynamics.state_dict(), filename + "_dynamics")