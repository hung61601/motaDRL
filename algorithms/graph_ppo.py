import torch
from torch.distributions.categorical import Categorical
from algorithms.policy.graph_isomorphism import GIN
from algorithms.buffer import RolloutBuffer
from algorithms.ppo import PPO, evaluate_action


class GraphPPO(PPO):
    def __init__(self,
                 embedding_policy: GIN,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 k_epochs: int = 3,
                 clip_ratio: float = 0.2,
                 entropy_coefficient: float = 0.01,
                 value_loss_coefficient: float = 0.5,
                 lr_decay_step_size: int = -1,
                 lr_decay_ratio: float = 1.0,
                 device: torch.device | str = 'auto'
                 ):
        """
        使用圖同構網路 (Graph Isomorphic Network)進行狀態嵌入，再透過近端策略優化演算法 (Proximal Policy Optimization) 學習策略。
        :param embedding_policy: 用於狀態嵌入的圖同構網路。
        :param learning_rate: 學習率。
        :param gamma: 0~1，獎勵的折扣因子 (discount factor)。
        :param k_epochs: 當策略更新時，對收集到的樣本使用 k_epochs 次來進行更新。
        :param clip_ratio: 裁切比率。當新舊策略的變化過大時，限制其更新的幅度。
        :param entropy_coefficient: 計算策略熵 (Policy Entropy) 的係數。係數越高時有助於提高策略的探索性，
            使得代理有更多的機會嘗試新的動作。
        :param value_loss_coefficient: 計算 Critic 網絡 (價值函數) 的損失函數係數。
        :param lr_decay_step_size: 策略網路每訓練 step_size 次之後調整一次學習率，預設 -1 不啟用。
        :param lr_decay_ratio: 調整學習率時衰減的比率。
        :param device: 運行程式碼所使用的設備 (cpu 或 cuda)。
        """
        super(GraphPPO, self).__init__(policy=embedding_policy,
                                       learning_rate=learning_rate,
                                       gamma=gamma,
                                       k_epochs=k_epochs,
                                       clip_ratio=clip_ratio,
                                       entropy_coefficient=entropy_coefficient,
                                       value_loss_coefficient=value_loss_coefficient,
                                       lr_decay_step_size=lr_decay_step_size,
                                       lr_decay_ratio=lr_decay_ratio,
                                       unfixed_number_action_space=True,
                                       device=device)

    def _evaluate(self, buffer: RolloutBuffer):
        action_log_probs = []
        state_values = []
        entropies = []
        for i in range(len(buffer.node_features)):
            action_prob, state_value = self.policy(node_features=buffer.node_features[i],
                                                   adj_matrix=buffer.adjacencies[i],
                                                   candidate=buffer.candidates[i],
                                                   mask=buffer.masks[i])
            action_log_prob, entropy = evaluate_action(action_prob.squeeze(), buffer.actions[i])
            action_log_probs.append(action_log_prob)
            state_values.append(state_value)
            entropies.append(entropy)
        action_log_probs = torch.stack(action_log_probs).to(self.device)
        state_values = torch.stack(state_values).to(self.device)
        entropies = torch.stack(entropies).to(self.device)
        return action_log_probs, state_values, entropies

    def _convert_tensor(self, observation: dict):
        fea_tensor = torch.tensor(observation['node_feature'], dtype=torch.float32).to(self.device)
        adj_tensor = torch.sparse_coo_tensor(
            indices=torch.tensor([observation['adj_matrix']['row'], observation['adj_matrix']['col']]),
            values=torch.ones(observation['adj_matrix']['value']),
            size=torch.Size(observation['adj_matrix']['size']),
            dtype=torch.float32).to(self.device)
        cand_tensor = torch.tensor(observation['candidate']).unsqueeze(-1).to(self.device)
        mask_tensor = torch.tensor(observation['mask']).unsqueeze(-1).to(self.device)
        return fea_tensor, adj_tensor, cand_tensor, mask_tensor

    def sample_action(self, observation: dict):
        fea_tensor, adj_tensor, cand_tensor, mask_tensor = self._convert_tensor(observation)
        with torch.no_grad():
            action_prob, _ = self.old_policy(node_features=fea_tensor,
                                             adj_matrix=adj_tensor,
                                             candidate=cand_tensor,
                                             mask=mask_tensor)
            dist = Categorical(action_prob.squeeze())
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        return fea_tensor, adj_tensor, cand_tensor, mask_tensor, action, action_log_prob

    def greedy_action(self, observation: dict):
        fea_tensor, adj_tensor, cand_tensor, mask_tensor = self._convert_tensor(observation)
        with torch.no_grad():
            action_prob, _ = self.old_policy(node_features=fea_tensor,
                                             adj_matrix=adj_tensor,
                                             candidate=cand_tensor,
                                             mask=mask_tensor)
            _, action = action_prob.squeeze().max(0)
        return action
