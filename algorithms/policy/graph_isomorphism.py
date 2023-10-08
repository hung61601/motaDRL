import torch
import torch.nn as nn
from algorithms.policy.actor_critic import ActorCritic


def build_module(input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int) -> list:
    if hidden_layers > 0:
        module_list = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ]
        for i in range(hidden_layers - 1):
            module_list.append(nn.Linear(hidden_dim, hidden_dim))
            module_list.append(nn.BatchNorm1d(hidden_dim))
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(hidden_dim, output_dim))
        module_list.append(nn.BatchNorm1d(output_dim))
        module_list.append(nn.ReLU())
    else:
        module_list = [
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        ]
    return module_list


def prepare_input_state(node_features: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    graph_pooled = node_features.clone().mean(dim=0, keepdim=True)
    gather_index = candidate.expand(-1, node_features.size(-1))
    candidate_feature = torch.gather(node_features, 0, gather_index)
    graph_pooled_repeated = graph_pooled.expand_as(candidate_feature)
    candidate_concat_feature = torch.cat((candidate_feature, graph_pooled_repeated), dim=-1)
    return candidate_concat_feature, graph_pooled


class GIN(nn.Module):
    def __init__(self,
                 policy: ActorCritic,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 mlp_hidden_layers: int,
                 mlp_block_layers: int,
                 device: torch.device | str,
                 aggregate_type: str = 'none'
                 ):
        super(GIN, self).__init__()
        self.policy = policy
        self.mlp_blocks_layers = mlp_block_layers
        self.device = device
        self.aggregate_type = aggregate_type
        self.mlp_blocks = torch.nn.ModuleList()
        if mlp_block_layers == 1:
            self.mlp_blocks.append(nn.Sequential(*build_module(
                input_dim, hidden_dim, output_dim, mlp_hidden_layers)).to(device))
        elif mlp_block_layers >= 2:
            self.mlp_blocks.append(nn.Sequential(*build_module(
                input_dim, hidden_dim, hidden_dim, mlp_hidden_layers)).to(device))
            for layer in range(mlp_block_layers - 2):
                self.mlp_blocks.append(nn.Sequential(*build_module(
                    hidden_dim, hidden_dim, hidden_dim, mlp_hidden_layers)).to(device))
            self.mlp_blocks.append(nn.Sequential(*build_module(
                hidden_dim, hidden_dim, output_dim, mlp_hidden_layers)).to(device))

    def forward(self,
                node_features: torch.Tensor,
                adj_matrix: torch.Tensor,
                candidate: torch.Tensor,
                mask: torch.Tensor
                ):
        for layer in range(self.mlp_blocks_layers):
            # 聚合鄰居節點的資訊
            if self.aggregate_type == 'sum':
                node_features = torch.mm(adj_matrix, node_features)
            elif self.aggregate_type == 'average':
                degree = torch.ones((adj_matrix.size(0), 1)).to(self.device)
                degree = torch.mm(adj_matrix, degree)
                node_features = torch.mm(adj_matrix, node_features) / degree
            # 獲得相鄰節點和中心節點的隱藏表示
            node_features = self.mlp_blocks[layer](node_features)
        # 重新塑形向量以便能輸入到後續的網路
        input_actor, input_critic = prepare_input_state(node_features, candidate,)
        action_probs, state_value = self.policy(input_actor, input_critic, mask)
        return action_probs, state_value
