import pytest
import math
import torch
import torch.nn as nn
from algorithms.policy.actor_critic import MLP, ActorCritic


class ReluMLP(nn.Module):
    def __init__(self):
        super(ReluMLP, self).__init__()
        self.fully_connected_1 = nn.Linear(2, 4)
        self.fully_connected_2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fully_connected_1(x))
        x = self.fully_connected_2(x)
        return x


class BeeEnv:
    """
    目標：讓小蜜蜂飛成 sin 圖形的軌道。
    # 小蜜蜂每次行動後增加 1 step，step 起始為 0 終止為 100
    # 軌道 target_y = sin(step * 3.14159 / 10)，震幅 1，週期 20
    # 小蜜蜂當前位置為 (step, y)
    # 小蜜蜂有四個動作 [y+0.3, y+0.1, y-0.3, y-0.1]
    # 每一個 step 能獲得的獎勵值為 1 - y
    # 每個動作的特徵為移動後的位置，全局特徵為所有移動後的位置的平均
    """
    def __init__(self, terminated_step, unfixed_number_action_space=False):
        self.terminated_step = terminated_step
        self.unfixed_number_action_space = unfixed_number_action_space
        self.step_count = 0
        self.agent_pos = [0, 0]
        self.target_y = [math.sin(i * math.pi / 10) for i in range(terminated_step + 1)]

    def get_state(self):
        if self.unfixed_number_action_space and self.step_count % 2 == 1:
            input_actor = torch.tensor([
                [self.step_count / self.terminated_step, self.agent_pos[1] + 0.3],
                [self.step_count / self.terminated_step, self.agent_pos[1] + 0.2],
                [self.step_count / self.terminated_step, self.agent_pos[1] + 0.1],
                [self.step_count / self.terminated_step, self.agent_pos[1] - 0.3],
                [self.step_count / self.terminated_step, self.agent_pos[1] - 0.2],
                [self.step_count / self.terminated_step, self.agent_pos[1] - 0.1]
            ], dtype=torch.float32)
            mask = torch.tensor([
                [False],
                [True],
                [False],
                [False],
                [False],
                [False]
            ])
        else:
            input_actor = torch.tensor([
                [self.step_count / self.terminated_step, self.agent_pos[1] + 0.3],
                [self.step_count / self.terminated_step, self.agent_pos[1] + 0.1],
                [self.step_count / self.terminated_step, self.agent_pos[1] - 0.3],
                [self.step_count / self.terminated_step, self.agent_pos[1] - 0.1]
            ], dtype=torch.float32)
            mask = torch.tensor([
                [False],
                [False],
                [False],
                [False]
            ])
        input_critic = torch.tensor([
            [self.step_count / self.terminated_step, self.agent_pos[1]]
        ], dtype=torch.float32)
        return input_actor, input_critic, mask

    def step(self, action):
        self.agent_pos[0] += 1
        if self.unfixed_number_action_space and self.step_count % 2 == 1:
            self.agent_pos[1] += (0.3, 0.2, 0.1, -0.3, -0.2, -0.1)[action]
        else:
            self.agent_pos[1] += (0.3, 0.1, -0.3, -0.1)[action]
        self.step_count += 1
        reward = 1.0 - abs(self.target_y[self.step_count] - self.agent_pos[1])
        return reward

    def reset(self):
        self.step_count = 0
        self.agent_pos = [0, 0]


def test_mlp_convergence():
    torch.manual_seed(600)
    # 建立訓練集
    train_x = torch.tensor([[10, 20], [0, 10], [50, 0], [0, 35], [40, 5]], dtype=torch.float32)
    # 標籤 y = (x1 / 10) - (x2 / 5)
    train_y = torch.tensor([[-3], [-2], [5], [-7], [3]], dtype=torch.float32)
    # 建立策略網路
    policy = MLP(2, 4, 1, 1)
    # 定義損失函數
    loss_function = nn.MSELoss()
    # 定義優化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

    last_loss = None
    for i_update in range(500):
        # 前向傳播
        output = policy(train_x)
        target = train_y
        # 計算損失
        loss = loss_function(output, target)
        # 斷言：損失應該會越來越小
        if last_loss is not None:
            volatility = 0.02
            assert loss < last_loss + volatility
        last_loss = loss.detach()
        # 清零梯度
        optimizer.zero_grad()
        # 反向傳播計算梯度
        loss.backward()
        # 使用優化器更新權重
        optimizer.step()

    # 訓練完成後，預測值應該相近於標籤
    assert train_y.detach() == pytest.approx(policy(train_x).detach(), 0.05)
    test_x = torch.tensor([[35, -10]], dtype=torch.float32)
    test_y = torch.tensor([[5.5]], dtype=torch.float32)
    assert test_y.detach() == pytest.approx(policy(test_x).detach(), 0.05)


def test_update_parameters():
    torch.manual_seed(600)
    train_x = torch.tensor([[10, 20], [0, 10], [50, 0], [0, 35], [40, 5]], dtype=torch.float32)
    train_y = torch.tensor([[-3], [-2], [5], [-7], [3]], dtype=torch.float32)
    policy = ReluMLP()
    loss_function = nn.MSELoss()

    last_loss = None
    for i_update in range(400):
        output = policy(train_x)
        target = train_y
        loss = loss_function(output, target)
        if last_loss is not None:
            assert loss < last_loss
        last_loss = loss.detach()
        # 清零梯度
        policy.zero_grad()
        # 反向傳播計算梯度
        loss.backward()
        # 查看梯度
        # print([f.grad.detach() for f in policy.parameters()])
        # 使用隨機梯度下降(SGD)更新權重
        learning_rate = 0.004
        for f in policy.parameters():
            f.data.sub_(f.grad.data * learning_rate)

    assert train_y.detach() == pytest.approx(policy(train_x).detach(), 0.1)
    test_x = torch.tensor([[35, -10]], dtype=torch.float32)
    test_y = torch.tensor([[5.5]], dtype=torch.float32)
    assert test_y.detach() == pytest.approx(policy(test_x).detach(), 0.1)


def test_build_module():
    input_actor = torch.tensor([
        [0.0, 0.3],
        [0.4, 0.1],
        [-0.2, -0.3],
        [0.8, -0.1]
    ], dtype=torch.float32)
    input_critic = torch.tensor([
        [0.5, 0.5]
    ], dtype=torch.float32)
    mask = torch.tensor([
        [False],
        [False],
        [False],
        [False]
    ])
    torch.manual_seed(600)
    sequential = ActorCritic(input_dim_actor=2,
                             hidden_dim_actor=10,
                             hidden_layers_actor=1,
                             input_dim_critic=2,
                             hidden_dim_critic=10,
                             hidden_layers_critic=1,
                             device=torch.device('cpu'))
    action_probs_1, state_value_1 = sequential(input_actor=input_actor,
                                               input_critic=input_critic,
                                               mask=mask)
    torch.manual_seed(600)
    module = ActorCritic(input_dim_actor=2,
                         hidden_dim_actor=10,
                         hidden_layers_actor=1,
                         input_dim_critic=2,
                         hidden_dim_critic=10,
                         hidden_layers_critic=1,
                         device=torch.device('cpu'))
    action_probs_2, state_value_2 = module(input_actor=input_actor,
                                           input_critic=input_critic,
                                           mask=mask)
    assert action_probs_1.equal(action_probs_2)
    assert state_value_1.equal(state_value_2)


def test_q_network():
    """
    此範例可以很好了解到超參數的設置對訓練結果的變化。
    hidden_dim 設為 4 時，訓練欠擬合 (under fitting) bias
    hidden_layers 小於 3 時也會欠擬合
    lr 學習率 0.01 最合適，過大或過小都使 loss 較高
    """
    torch.manual_seed(600)
    num_envs = 4
    episode = 50
    k_epochs = 40
    terminated_step = 20
    policy = MLP(2, 10, 1, 3)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    all_episode_rewards = []
    env = BeeEnv(terminated_step)
    for i_update in range(episode):
        # 建立訓練集
        train_x = []
        train_y = []
        for _ in range(num_envs):
            env.reset()
            for step in range(terminated_step):
                _, input_critic, _ = env.get_state()
                action_dist = torch.distributions.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
                action = action_dist.sample().item()
                reward = env.step(action)
                train_x.append(input_critic)
                train_y.append(reward)
        # 調整訓練集形狀，!!!注意!!! 形狀不正確會導致訓練無法收斂，train_x 和 train_y 都要
        correct_shape_x = torch.stack(train_x)
        correct_shape_y = torch.tensor(train_y, dtype=torch.float32)
        correct_shape_x = correct_shape_x.squeeze()     # Tensor(80, 1, 2) -> Tensor(80, 2)
        correct_shape_y = correct_shape_y.unsqueeze(1)  # Tensor(80,)      -> Tensor(80, 1)
        for _ in range(k_epochs):
            output = policy(correct_shape_x)
            target = correct_shape_y
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 驗證結果
        env.reset()
        episode_reward = []
        for step in range(terminated_step):
            input_x, _, _ = env.get_state()
            output_y = policy(input_x)
            _, action = output_y.squeeze().max(0)
            reward = env.step(action.item())
            episode_reward.append(reward)
        all_episode_rewards.append(sum(episode_reward))
    # 斷言：累積獎勵應呈上升趨勢 (學到策略)
    half = len(all_episode_rewards) // 2
    first_half_performance = sum(all_episode_rewards[:half])
    second_half_performance = sum(all_episode_rewards[half:])
    assert second_half_performance > first_half_performance * 1.5
    # 斷言：累積獎勵應接近滿分成績
    assert second_half_performance > half * terminated_step * 0.7


def test_actor():
    torch.manual_seed(600)
    num_envs = 4
    episode = 50
    terminated_step = 20
    policy = MLP(2, 8, 1, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.08)
    all_episode_rewards = []
    env = BeeEnv(terminated_step)
    for i_update in range(episode):
        # 收集 rollout 資料
        old_states = []
        old_actions = []
        old_rewards = []
        for _ in range(num_envs):
            env.reset()
            for step in range(terminated_step):
                with torch.no_grad():
                    input_actor, _, _ = env.get_state()
                    action_probs = torch.softmax(policy(input_actor), dim=-2).squeeze()
                    # 採樣動作
                    # 依權重採樣：torch.multinomial(weights, 2)  weights = [0, 10, 3, 0]
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                    reward = env.step(action)
                    old_states.append(input_actor)
                    old_actions.append(action)
                    old_rewards.append(reward)
        # 開始訓練，產生計算圖，使用梯度上升法來更新 Actor 網絡的參數，以最大化回報
        losses = []
        for input_actor, action, reward in zip(old_states, old_actions, old_rewards):
            # 不能使用 rollout 的 action_probs，會無法收斂
            action_probs = torch.softmax(policy(input_actor), dim=-2).squeeze()
            # reward 的尺度縮放不影響訓練
            reward *= 123
            losses.append(-torch.log(action_probs[action]) * reward)
        optimizer.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        # 驗證結果
        env.reset()
        episode_reward = []
        for step in range(terminated_step):
            input_x, _, _ = env.get_state()
            output_y = policy(input_x)
            _, action = output_y.squeeze().max(0)
            reward = env.step(action.item())
            episode_reward.append(reward)
        all_episode_rewards.append(sum(episode_reward))
    # 斷言：累積獎勵應呈上升趨勢 (學到策略)
    half = len(all_episode_rewards) // 2
    first_half_performance = sum(all_episode_rewards[:half])
    second_half_performance = sum(all_episode_rewards[half:])
    assert second_half_performance > first_half_performance * 1.3
    # 斷言：累積獎勵應接近滿分成績
    assert second_half_performance > half * terminated_step * 0.6


def test_critic():
    torch.manual_seed(600)
    num_envs = 4
    episode = 50
    terminated_step = 20
    policy = MLP(2, 10, 1, 2)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
    all_episode_rewards = []
    env = BeeEnv(terminated_step)
    for i_update in range(episode):
        # 收集 rollout 資料
        old_states = []
        old_rewards = []
        for _ in range(num_envs):
            env.reset()
            for step in range(terminated_step):
                with torch.no_grad():
                    input_actor, input_critic, _ = env.get_state()
                    action_probs = torch.softmax(policy(input_actor), dim=-2).squeeze()
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
                    reward = env.step(action)
                    old_states.append(input_critic)
                    old_rewards.append(reward)
        # 開始訓練
        state_values = policy(torch.stack(old_states))
        loss = loss_function(state_values.squeeze(), torch.tensor(old_rewards))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 驗證結果
        env.reset()
        episode_reward = []
        for step in range(terminated_step):
            input_x, _, _ = env.get_state()
            output_y = policy(input_x)
            _, action = output_y.squeeze().max(0)
            reward = env.step(action.item())
            episode_reward.append(reward)
        all_episode_rewards.append(sum(episode_reward))
    # 斷言：累積獎勵應呈上升趨勢 (學到策略)
    half = len(all_episode_rewards) // 2
    first_half_performance = sum(all_episode_rewards[:half])
    second_half_performance = sum(all_episode_rewards[half:])
    assert second_half_performance > first_half_performance * 1.3
    # 斷言：累積獎勵應接近滿分成績
    assert second_half_performance > half * terminated_step * 0.5


def test_actor_critic():
    """
    本示例展示了價值網路和策略網路之間如何搭配，並表現出超越單使用其中一種網路的高效能。
    """
    torch.manual_seed(600)
    num_envs = 4
    episode = 50
    terminated_step = 20
    actor_network = MLP(2, 8, 1, 1)
    critic_network = MLP(2, 10, 1, 3)
    actor_optimizer = torch.optim.Adam(actor_network.parameters(), lr=0.08)
    critic_optimizer = torch.optim.Adam(critic_network.parameters(), lr=0.01)
    value_loss_function = nn.MSELoss()
    all_episode_rewards = []
    env = BeeEnv(terminated_step)
    for i_update in range(episode):
        # 收集資料
        old_input_actor = []
        old_input_critic = []
        old_actions = []
        old_rewards = []
        for _ in range(num_envs):
            env.reset()
            for step in range(terminated_step):
                with torch.no_grad():
                    input_actor, input_critic, _ = env.get_state()
                    action_prob = torch.softmax(actor_network(input_actor), dim=-2)
                    action_dist = torch.distributions.Categorical(action_prob.squeeze())
                    action = action_dist.sample().item()
                    reward = env.step(action)
                    old_input_actor.append(input_actor)
                    old_input_critic.append(input_critic)
                    old_actions.append(action)
                    old_rewards.append(reward)
        # 計算折扣後的獎勵
        # 該任務不適合用太高的折扣因子
        # 當為 0 時依然能表現出好的效能
        # 當為 1 時會出現災難式結果
        rewards = []
        discount_factor = 0.1
        discounted_reward = 0
        for reward in old_rewards[::-1]:
            discounted_reward = reward + discount_factor * discounted_reward
            rewards.insert(0, discounted_reward)
        returns = torch.tensor(rewards)
        # 進行 Z-score 標準化 (可有可無)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # 計算價值網路的損失
        state_values = critic_network(torch.stack(old_input_critic))
        critic_loss = value_loss_function(state_values.squeeze(), returns)  # 梯度下降
        # actor 使用 critic 提供的價值信息，例如優勢值 (advantage)
        advantages = returns - state_values.view(-1).detach()
        # 計算策略網路的損失
        actor_losses = []
        for input_actor, action, reward in zip(old_input_actor, old_actions, old_rewards):
            action_prob = torch.softmax(actor_network(input_actor), dim=-2)
            actor_losses.append(torch.log(action_prob.squeeze()[action]))
        actor_loss = -(torch.stack(actor_losses) * advantages).mean()  # 梯度上升 (轉換正負號)
        # 更新策略網路和價值網路
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()
        # 驗證結果
        env.reset()
        episode_reward = []
        for step in range(terminated_step):
            input_x, _, _ = env.get_state()
            output_y = actor_network(input_x)
            _, action = output_y.squeeze().max(0)
            reward = env.step(action.item())
            episode_reward.append(reward)
        all_episode_rewards.append(sum(episode_reward))
        print(sum(episode_reward))
    # 斷言：累積獎勵應呈上升趨勢 (學到策略)
    half = len(all_episode_rewards) // 2
    first_half_performance = sum(all_episode_rewards[:half]) / half / terminated_step
    second_half_performance = sum(all_episode_rewards[half:]) / half / terminated_step
    assert second_half_performance > first_half_performance + 1


@pytest.mark.skip(reason="incorrect test")
def test_actor_critic_unable_to_converge():
    """
    這是錯誤版本。
    """
    torch.manual_seed(600)
    episode = 50
    terminated_step = 20
    policy = ActorCritic(input_dim_actor=2,
                         hidden_dim_actor=10,
                         hidden_layers_actor=1,
                         input_dim_critic=2,
                         hidden_dim_critic=10,
                         hidden_layers_critic=1,
                         device=torch.device('cpu'))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    all_episode_rewards = []
    env = BeeEnv(terminated_step)
    for _ in range(episode):
        env.reset()
        episode_action_log_probs = []
        episode_state_values = []
        episode_rewards = []
        for step in range(terminated_step):
            input_actor, input_critic, mask = env.get_state()
            # 選擇動作
            action_probs, state_value = policy(input_actor=input_actor,
                                               input_critic=input_critic,
                                               mask=mask)
            action_dist = torch.distributions.Categorical(action_probs.squeeze())
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)
            # 更新狀態並獲得獎勵
            reward = env.step(action.item())
            # 儲存動作機率的對數、狀態和獎勵
            episode_action_log_probs.append(action_log_prob)
            episode_state_values.append(state_value)
            episode_rewards.append(reward)
        # 計算折扣後的獎勵，用於評估一個策略或行動序列的效能指標
        returns = []
        discount_factor = 0.99
        discounted_reward = 0
        for reward in episode_rewards[::-1]:
            discounted_reward = reward + discount_factor * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        # 計算損失
        policy_losses = []
        value_losses = []
        # actor (policy) loss: 當選澤動作的機率越高越確定，loss 就越低
        for log_prob, ret in zip(episode_action_log_probs, returns):
            policy_losses.append(-log_prob * ret.item())
        # critic (value) loss: 當預測的價值與實際價值越接近，loss 就越低
        for value, ret in zip(episode_state_values, returns):
            value_losses.append(loss_function(value, ret))
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        # 策略更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 清空資料
        all_episode_rewards.append(sum(episode_rewards))
        del episode_action_log_probs[:]
        del episode_state_values[:]
        del episode_rewards[:]
    # 斷言：累積獎勵無法有效上升
    half = len(all_episode_rewards) // 2
    first_half_performance = sum(all_episode_rewards[:half]) / half / terminated_step
    second_half_performance = sum(all_episode_rewards[half:]) / half / terminated_step
    assert not second_half_performance > first_half_performance + 0.1
