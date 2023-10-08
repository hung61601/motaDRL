import copy
import torch
from algorithms.policy.graph_isomorphism import GIN
from algorithms.policy.actor_critic import ActorCritic
from algorithms.graph_ppo import GraphPPO
from env.util import get_device
from env.mota_env import Mota
from env.mota_builder import MotaBuilder
from env.logger import Logger
from env import validation_data


def validation(env: Mota, mota_builder: MotaBuilder, model: GraphPPO, logger: Logger, episode: int) -> int:
    observation, _ = env.reset(*copy.deepcopy(mota_builder.build()))
    terminated = truncated = False
    while not terminated and not truncated:
        if len(observation['candidate']) > 1:
            action = model.greedy_action(observation)
            observation, _, terminated, truncated, _ = env.step(action.item())
        else:
            observation, _, terminated, truncated, _ = env.step(0)
    logger.validation_log(episode, env.score)
    return env.score


if __name__ == '__main__':
    device = get_device()
    print('Device set to:', device)
    torch.manual_seed(600)
    # settings
    node_feature_size = 9
    gin_output_dim = 64
    model_folder_name = '../models/GraphPPO/2023-10-07-163419/'
    model_file_name = '5000_2955.pth'
    mota_env = Mota(graphic_depth=5, use_advanced_feature=True)
    builder = validation_data.map02()
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
    mota_model = GraphPPO(embedding_policy=embedding_policy,
                          learning_rate=2e-3,
                          gamma=1.0,
                          k_epochs=1,
                          clip_ratio=0.2,
                          entropy_coefficient=0.01,
                          value_loss_coefficient=0.5,
                          lr_decay_step_size=-1,
                          lr_decay_ratio=1.0,
                          device=device)
    mota_logger = Logger(model_folder_name=model_folder_name, write_info_message=None, write=False)
    mota_logger.load_model(file_name=model_file_name, model=mota_model)
    validation_score = validation(env=mota_env,
                                  mota_builder=builder,
                                  model=mota_model,
                                  logger=mota_logger,
                                  episode=0)
