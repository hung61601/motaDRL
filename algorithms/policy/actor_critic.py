import torch
import torch.nn as nn
import torch.nn.functional as F


def build_module(input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int) -> list:
    if hidden_layers > 0:
        module_list = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for i in range(hidden_layers - 1):
            module_list.append(nn.Linear(hidden_dim, hidden_dim))
            module_list.append(nn.Tanh())
        module_list.append(nn.Linear(hidden_dim, output_dim))
    else:
        module_list = [nn.Linear(input_dim, output_dim)]
    return module_list


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 hidden_layers: int
                 ):
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers
        self.multi_linear = torch.nn.ModuleList()
        if hidden_layers > 0:
            self.multi_linear.append(nn.Linear(input_dim, hidden_dim))
            for i in range(hidden_layers - 1):
                self.multi_linear.append(nn.Linear(hidden_dim, hidden_dim))
            self.multi_linear.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.multi_linear.append(nn.Linear(input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in range(self.hidden_layers):
            x = torch.tanh(self.multi_linear[layer](x))
        return self.multi_linear[self.hidden_layers](x)


class ActorCritic(nn.Module):
    def __init__(self,
                 input_dim_actor: int,
                 hidden_dim_actor: int,
                 hidden_layers_actor: int,
                 input_dim_critic: int,
                 hidden_dim_critic: int,
                 hidden_layers_critic: int,
                 device: torch.device,
                 use_sequential: bool = False
                 ):
        super(ActorCritic, self).__init__()
        if use_sequential:
            self.actor = nn.Sequential(*build_module(
                input_dim_actor, hidden_dim_actor, 1, hidden_layers_actor)).to(device)
            self.critic = nn.Sequential(*build_module(
                input_dim_critic, hidden_dim_critic, 1, hidden_layers_critic)).to(device)
        else:
            self.actor = MLP(
                input_dim_actor, hidden_dim_actor, 1, hidden_layers_actor).to(device)
            self.critic = MLP(
                input_dim_critic, hidden_dim_critic, 1, hidden_layers_critic).to(device)

    def forward(self,
                input_actor: torch.Tensor,
                input_critic: torch.Tensor,
                mask: torch.Tensor
                ):
        scores = self.actor(input_actor)
        scores[mask] = float('-inf')
        action_probs = F.softmax(scores, dim=-2)
        state_value = self.critic(input_critic)
        return action_probs, state_value
