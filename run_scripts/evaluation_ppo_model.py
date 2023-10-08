import copy
import torch
from algorithms.ppo import PPO
from algorithms.policy.actor_critic import ActorCritic
from env.util import get_device
from env.mota_env import Mota
from env.logger import Logger
from env import validation_data
from run_scripts.training_ppo_model import convert_state


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

    logger = Logger(model_folder_name=model_folder_name, write_info_message=None, write=False)
    logger.load_model(file_name=model_file_name, model=model)

    env = Mota(graphic_depth=5, use_advanced_feature=True)
    mota_builder = validation_data.map02()
    mota_data = mota_builder.build()

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
    logger.validation_log(0, env.score)
