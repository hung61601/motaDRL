import copy
import torch
import datetime
from algorithms.graph_ppo import GraphPPO
from algorithms.buffer import RolloutBuffer
from algorithms.policy.graph_isomorphism import GIN
from algorithms.policy.actor_critic import ActorCritic
from env.util import get_device
from env.mota_env import Mota
from env import validation_data
from env.logger import Logger
from run_scripts.evaluation_graph_ppo_model import validation


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
    gin_output_dim = 64
    normalize_reward_scale = 0.01
    # if you want to continue training the model, please fill in the following parameters
    continue_training = False
    model_folder_name = '../models/GraphPPO/2023-10-07-163419/'
    model_file_name = '5000_2955.pth'

    # build policy network
    policy = ActorCritic(input_dim_actor=gin_output_dim * 2,
                         hidden_dim_actor=64,
                         hidden_layers_actor=1,
                         input_dim_critic=gin_output_dim,
                         hidden_dim_critic=64,
                         hidden_layers_critic=1,
                         device=device)
    embedding_policy = GIN(policy=policy,
                           input_dim=node_feature_size,
                           hidden_dim=64,
                           output_dim=64,
                           mlp_hidden_layers=1,
                           mlp_block_layers=2,
                           device=device,
                           aggregate_type='sum')
    model = GraphPPO(embedding_policy=embedding_policy,
                     learning_rate=2e-3,
                     gamma=1.0,
                     k_epochs=1,
                     clip_ratio=0.2,
                     entropy_coefficient=0.01,
                     value_loss_coefficient=0.5,
                     lr_decay_step_size=-1,
                     lr_decay_ratio=1.0,
                     device=device)

    # load model
    if continue_training:
        logger = Logger(model_folder_name=model_folder_name, write_info_message=None)
        episode = logger.load_model(file_name=model_file_name, model=model)
    else:
        logger = Logger(model_folder_name=f'../models/GraphPPO/{datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")}/')
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
                    fea_tensor, adj_tensor, cand_tensor, mask_tensor, action, action_log_prob =\
                        model.sample_action(observation)
                    observation, reward, terminated, truncated, _ = env.step(action.item())
                    is_terminal = terminated or truncated
                    reward *= normalize_reward_scale
                    # saving episode data
                    buffer.add(action=action,
                               node_feature=fea_tensor,
                               graph_feature=None,
                               adjacency=adj_tensor,
                               candidate=cand_tensor,
                               mask=mask_tensor,
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
            validation_score = validation(env=env,
                                          mota_builder=mota_builder,
                                          model=model,
                                          logger=logger,
                                          episode=episode)
            # save model
            logger.save_model(episode=episode,
                              file_name=f'{episode}_{validation_score}.pth',
                              model=model)
