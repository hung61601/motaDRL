import torch
import pytest
from algorithms.ppo import evaluate_action, PPO
from algorithms.buffer import RolloutBuffer
from algorithms.policy.actor_critic import ActorCritic
from actor_critic_test import BeeEnv


def test_evaluate_action():
    probability = torch.tensor([0.25, 0.25, 0.5])
    log_probability, entropy = evaluate_action(probability, torch.tensor(0))
    assert log_probability.item() == pytest.approx(-1.386294)
    assert entropy.item() == pytest.approx(1.039720)
    log_probability, entropy = evaluate_action(probability, torch.tensor(2))
    assert log_probability.item() == pytest.approx(-0.693147)
    assert entropy.item() == pytest.approx(1.039720)
    probability = torch.tensor([0.1, 0.9])
    log_probability, entropy = evaluate_action(probability, torch.tensor(0))
    assert log_probability.item() == pytest.approx(-2.302585)
    assert entropy.item() == pytest.approx(0.3250829)


def test_ppo_update():
    """
    該任務的最佳訓練回數：
    num_envs = 4
    episode = 500
    k_epochs = 10
    """
    torch.manual_seed(600)
    num_envs = 4
    episode = 10
    terminated_step = 20
    policy = ActorCritic(input_dim_actor=2,
                         hidden_dim_actor=10,
                         hidden_layers_actor=1,
                         input_dim_critic=2,
                         hidden_dim_critic=10,
                         hidden_layers_critic=1,
                         device=torch.device('cpu'))
    model = PPO(policy=policy,
                learning_rate=0.05,
                gamma=0.1,
                k_epochs=40,
                value_loss_coefficient=1,
                unfixed_number_action_space=False,
                device=torch.device('cpu'))

    buffers = [RolloutBuffer() for _ in range(num_envs)]
    env = BeeEnv(terminated_step, unfixed_number_action_space=False)
    for _ in range(episode):
        for buffer in buffers:
            env.reset()
            for step in range(terminated_step):
                input_actor, input_critic, mask = env.get_state()
                # 選擇動作
                with torch.no_grad():
                    action_probs, _ = model.old_policy(input_actor=input_actor,
                                                       input_critic=input_critic,
                                                       mask=mask)
                action_dist = torch.distributions.Categorical(action_probs.squeeze())
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                # 更新狀態並獲得獎勵
                reward = env.step(action.item())
                # 儲存動作機率的對數、狀態和獎勵
                buffer.add(action=action,
                           node_feature=input_actor,
                           graph_feature=input_critic,
                           adjacency=None,
                           candidate=None,
                           mask=mask,
                           reward=reward,
                           log_prob=action_log_prob,
                           state_value=None,
                           terminal=(step == terminated_step - 1))
        # 策略更新
        model.update(buffers)
        # 清空資料
        for buffer in buffers:
            buffer.clear()
    # 驗證結果
    env.reset()
    episode_reward = []
    for step in range(terminated_step):
        input_actor, input_critic, mask = env.get_state()
        with torch.no_grad():
            action_probs, _ = model.old_policy(input_actor=input_actor,
                                               input_critic=input_critic,
                                               mask=mask)
            _, action = action_probs.squeeze().max(0)
            reward = env.step(action.item())
            episode_reward.append(reward)
    assert sum(episode_reward) / terminated_step > 0.85
