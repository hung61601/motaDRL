import torch
from algorithms.policy.graph_isomorphism import build_module, prepare_input_state, GIN
from algorithms.policy.actor_critic import ActorCritic


def test_build_module():
    module_list = build_module(1, 1, 1, 0)
    assert len(module_list) == 3
    module_list = build_module(1, 1, 1, 1)
    assert len(module_list) == 6
    module_list = build_module(1, 1, 1, 3)
    assert len(module_list) == 12


def test_prepare_input_state():
    node_features = torch.tensor([[0.1, 0.5, -0.3], [1.1, -0.5, -1.2], [0.8, 1.5, 0.5], [0.0, 1.0, 0.0]])
    candidate = torch.tensor([[0], [3]])
    node_feature, graph_feature = prepare_input_state(node_features, candidate)
    assert node_feature.equal(torch.tensor([
        [0.1, 0.5, -0.3, 0.5, 0.625, -0.25],
        [0.0, 1.0,  0.0, 0.5, 0.625, -0.25]
    ]))
    assert graph_feature.equal(torch.tensor([
        [0.5, 0.625, -0.25]
    ]))


def test_gin_forward():
    torch.manual_seed(600)
    node_features = torch.tensor([
        [0.1, 0.5, -0.3],
        [1.1, -0.5, -1.2],
        [0.8, 1.5, 0.5],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    adj_matrix = torch.tensor([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ], dtype=torch.float32)
    candidate = torch.tensor([[0], [1], [3]])
    mask = torch.tensor([[True], [False], [False]])
    policy = ActorCritic(14, 10, 0, 7, 12, 0, torch.device('cpu'), True)
    sum_model = GIN(policy, 3, 4, 7, 2, 1, torch.device('cpu'), 'sum')
    action_probs, state_value = sum_model(node_features, adj_matrix, candidate, mask)
    assert action_probs.shape == (3, 1)
    assert state_value.shape == (1, 1)
    avg_model = GIN(policy, 3, 4, 7, 2, 1, torch.device('cpu'), 'average')
    action_probs, state_value = avg_model(node_features, adj_matrix, candidate, mask)
    assert action_probs.shape == (3, 1)
    assert state_value.shape == (1, 1)
