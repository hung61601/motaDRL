import copy
import torch
import datetime
from algorithms.ppo import PPO
from algorithms.buffer import RolloutBuffer
from algorithms.policy.actor_critic import ActorCritic
from torch.distributions.categorical import Categorical
from env.util import get_device
from env.mota_env import Mota
from env import validation_data
from env.logger import Logger


def convert_state(info, num_aggregate=0):
    h_nodes = torch.tensor(info['node_feature'], dtype=torch.float32)
    candidate = torch.tensor(info['candidate']).unsqueeze(-1)
    action_mask = torch.tensor(info['mask']).unsqueeze(-1)
    # aggregate
    adj_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor([info['adj_matrix']['row'], info['adj_matrix']['col']]),
        values=torch.ones(info['adj_matrix']['value']),
        size=torch.Size(info['adj_matrix']['size']),
        dtype=torch.float32)
    degree = torch.mm(adj_matrix, torch.ones((adj_matrix.size(0), 1)))
    for _ in range(num_aggregate):
        h_nodes = torch.mm(adj_matrix, h_nodes) / degree
    # prepare features
    h_pooled = h_nodes.clone().mean(dim=0, keepdim=True)
    feature_size = h_nodes.size(-1)
    gather_index = candidate.expand(-1, feature_size)
    candidate_feature = torch.gather(h_nodes, 0, gather_index)
    h_pooled_repeated = h_pooled.expand_as(candidate_feature)
    concat_feature = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
    return concat_feature, h_pooled, action_mask


if __name__ == '__main__':
    device = get_device()
    print('Device set to:', device)
    torch.manual_seed(600)
    # settings
    num_envs = 4
    num_episode = 5000
    log_frequency = 1
    validation_frequency = 10
    node_feature_size = 9
    normalize_reward_scale = 0.01
    # if you want to continue training the model, please fill in the following parameters
    continue_training = False
    model_folder_name = '../models/PPO/2023-10-07-222023/'
    model_file_name = '5000_2987.pth'

    # build policy network
    policy = ActorCritic(input_dim_actor=node_feature_size * 2,
                         hidden_dim_actor=64,
                         hidden_layers_actor=1,
                         input_dim_critic=node_feature_size,
                         hidden_dim_critic=64,
                         hidden_layers_critic=1,
                         device=device)
    model = PPO(policy=policy,
                learning_rate=2e-3,
                gamma=1.0,
                k_epochs=1,
                clip_ratio=0.2,
                entropy_coefficient=0.01,
                value_loss_coefficient=0.5,
                lr_decay_step_size=-1,
                lr_decay_ratio=1.0,
                unfixed_number_action_space=True,
                device=device)

    # load model
    if continue_training:
        logger = Logger(model_folder_name=model_folder_name, write_info_message=None)
        episode = logger.load_model(file_name=model_file_name, model=model)
    else:
        logger = Logger(model_folder_name=f'../models/PPO/{datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")}/')
        episode = 0

    # start training
    buffers = [RolloutBuffer() for _ in range(num_envs)]
    env = Mota(graphic_depth=5, use_advanced_feature=True)
    mota_builder = validation_data.map02()
    mota_data = mota_builder.build()

    # rollout the env
    for episode in range(episode + 1, num_episode + 1):
        ep_rewards = []
        for buffer in buffers:
            observation, _ = env.reset(*copy.deepcopy(mota_data))
            while True:
                if len(observation['candidate']) > 1:
                    input_actor, input_critic, mask = convert_state(observation, num_aggregate=2)
                    with torch.no_grad():
                        action_probs, _ = model.old_policy(input_actor=input_actor,
                                                           input_critic=input_critic,
                                                           mask=mask)
                        dist = Categorical(action_probs.squeeze())
                        action = dist.sample()
                        action_log_prob = dist.log_prob(action)
                    observation, reward, terminated, truncated, _ = env.step(action.item())
                    is_terminal = terminated or truncated
                    reward *= normalize_reward_scale
                    # saving episode data
                    buffer.add(action=action,
                               node_feature=input_actor,
                               graph_feature=input_critic,
                               adjacency=None,
                               candidate=None,
                               mask=mask,
                               reward=reward,
                               log_prob=action_log_prob,
                               state_value=None,
                               terminal=is_terminal)
                else:
                    observation, reward, terminated, truncated, _ = env.step(0)
                    is_terminal = terminated or truncated
                    reward *= normalize_reward_scale
                    buffer.rewards[-1] += reward
                    buffer.terminals[-1] = is_terminal
                if is_terminal:
                    break
            ep_rewards.append(sum(buffer.rewards))
        average_episode_reward = sum(ep_rewards) / len(ep_rewards)

        # update policy
        loss, value_loss = model.update(buffers)
        for buffer in buffers:
            buffer.clear()

        # log results
        if episode % log_frequency == 0:
            logger.training_log(episode=episode,
                                reward=average_episode_reward,
                                loss=loss,
                                value_loss=value_loss)
        # validation
        if episode % validation_frequency == 0:
            observation, _ = env.reset(*copy.deepcopy(mota_data))
            terminated = truncated = False
            while not terminated and not truncated:
                if len(observation['candidate']) > 1:
                    input_actor, input_critic, mask = convert_state(observation)
                    with torch.no_grad():
                        action_probs, _ = model.old_policy(input_actor=input_actor,
                                                           input_critic=input_critic,
                                                           mask=mask)
                        _, action = action_probs.squeeze().max(0)
                    observation, _, terminated, truncated, _ = env.step(action.item())
                else:
                    observation, _, terminated, truncated, _ = env.step(0)
            logger.validation_log(episode, env.score)
            # save model
            logger.save_model(episode=episode,
                              file_name=f'{episode}_{env.score}.pth',
                              model=model)
