import torch
import torch.nn as nn
from copy import deepcopy
from algorithms.buffer import RolloutBuffer
from algorithms.policy.actor_critic import ActorCritic
from torch.distributions.categorical import Categorical


def evaluate_action(action_probs, action):
    softmax_dist = Categorical(action_probs)
    log_prob = softmax_dist.log_prob(action)
    entropy = softmax_dist.entropy()
    return log_prob, entropy


class PPO:
    def __init__(self,
                 policy: ActorCritic | nn.Module,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 k_epochs: int = 3,
                 clip_ratio: float = 0.2,
                 entropy_coefficient: float = 0.01,
                 value_loss_coefficient: float = 0.5,
                 lr_decay_step_size: int = -1,
                 lr_decay_ratio: float = 1.0,
                 unfixed_number_action_space: bool = False,
                 device: torch.device | str = 'auto'
                 ):
        """
        近端策略優化演算法 (Proximal Policy Optimization)。
        :param policy: 用於學習的策略網路。例如由 MLP 組成的 Actor-Critic。
        :param learning_rate: 學習率。
        :param gamma: 0~1，獎勵的折扣因子 (discount factor)。
        :param k_epochs: 當策略更新時，對收集到的樣本使用 k_epochs 次來進行更新。
        :param clip_ratio: 裁切比率。當新舊策略的變化過大時，限制其更新的幅度。
        :param entropy_coefficient: 計算策略熵 (Policy Entropy) 的係數。係數越高時有助於提高策略的探索性，
            使得代理有更多的機會嘗試新的動作。
        :param value_loss_coefficient: 計算 Critic 網絡 (價值函數) 的損失函數係數。
        :param lr_decay_step_size: 策略網路每訓練 step_size 次之後調整一次學習率，預設 -1 不啟用。
        :param lr_decay_ratio: 調整學習率時衰減的比率。
        :param unfixed_number_action_space: 每個狀態的動作數量是否相同，預設 False 為相同。
        :param device: 運行程式碼所使用的設備 (cpu 或 cuda)。
        """
        self.policy = policy
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.unfixed_number_action_space = unfixed_number_action_space
        self.device = device
        self.old_policy = deepcopy(self.policy)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.scheduler = None
        if lr_decay_step_size != -1:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_decay_step_size, gamma=lr_decay_ratio)

    def _estimate_returns(self,
                          rewards: list,
                          terminals: list) -> list:
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return returns

    def _evaluate(self, buffer: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.unfixed_number_action_space:
            # TODO: 不堆疊向量就輸入神經網路，速度會慢上 2 倍以上，請優化計算速度
            action_log_probs = []
            state_values = []
            entropies = []
            for i in range(len(buffer.node_features)):
                action_prob, state_value = self.policy(input_actor=buffer.node_features[i],
                                                       input_critic=buffer.graph_features[i],
                                                       mask=buffer.masks[i])
                action_log_prob, entropy = evaluate_action(action_prob.squeeze(), buffer.actions[i])
                action_log_probs.append(action_log_prob)
                state_values.append(state_value)
                entropies.append(entropy)
            action_log_probs = torch.stack(action_log_probs).to(self.device)
            state_values = torch.stack(state_values).to(self.device)
            entropies = torch.stack(entropies).to(self.device)
        else:
            action_probs, state_values = self.policy(input_actor=torch.stack(buffer.node_features).to(self.device),
                                                     input_critic=torch.stack(buffer.graph_features).to(self.device),
                                                     mask=torch.stack(buffer.masks).to(self.device))
            action_log_probs, entropies = evaluate_action(action_probs.squeeze(),
                                                          torch.stack(buffer.actions).to(self.device))
        return action_log_probs, state_values, entropies

    def update(self,
               buffers: list[RolloutBuffer]) -> tuple[float, float]:
        log = {'loss': 0.0, 'value_loss': 0.0}
        buffers_returns = []
        buffers_action_log_probs = []
        for buffer in buffers:
            # 計算折現後的獎勵，用於評估一個策略或行動序列的效能指標
            returns = self._estimate_returns(buffer.rewards, buffer.terminals)
            # 轉換成向量
            buffers_returns.append(
                torch.tensor(returns, dtype=torch.float32).to(self.device))
            buffers_action_log_probs.append(
                torch.stack(buffer.log_probs).to(self.device).detach())
        # 多次優化策略
        for _ in range(self.k_epochs):
            loss_sum = torch.tensor(0.).to(self.device)
            for buffer, returns, old_action_log_probs in zip(buffers, buffers_returns, buffers_action_log_probs):
                # 評估舊動作和狀態值
                action_log_probs, state_values, entropies = self._evaluate(buffer)
                # 計算損失
                ratios = torch.exp(action_log_probs - old_action_log_probs)
                advantages = returns - state_values.view(-1).detach()
                surrogate_loss_1 = ratios * advantages
                surrogate_loss_2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surrogate_loss_1, surrogate_loss_2).mean()
                value_loss = self.loss_function(state_values.squeeze(), returns)
                entropy_loss = -entropies.mean()
                loss = (policy_loss +
                        self.value_loss_coefficient * value_loss +
                        self.entropy_coefficient * entropy_loss)
                log['value_loss'] += value_loss.item()
                log['loss'] += loss.item()
                loss_sum += loss
            # 策略更新
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()
            # 調整學習率
            if self.scheduler is not None:
                self.scheduler.step()
        # 複製新策略參數到舊策略
        self.old_policy.load_state_dict(self.policy.state_dict())
        size_number = len(buffers) * self.k_epochs
        return log['loss'] / size_number, log['value_loss'] / size_number
