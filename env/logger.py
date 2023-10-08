import os
import csv
import torch
import datetime
from algorithms.ppo import PPO


class Logger:
    def __init__(self, model_folder_name: str, write_info_message: str | None = None, write: bool = True):
        """
        負責處理模型的加載和儲存，以及打印訓練過程中的消息。
        :param model_folder_name: 模型所存放的資料夾名稱。
        :param write_info_message: 額外紀錄的訊息。
        :param write: 是否啟用訊息紀錄。
        """
        self.model_folder_name = model_folder_name
        self.write = write
        if not os.path.exists(model_folder_name):
            os.makedirs(model_folder_name)
            header = ['Datetime', 'Episode', 'Reward', 'Loss', 'Value_Loss']
            self.write_log(model_folder_name + '!training_log.csv', header)
            header = ['Datetime', 'Episode', 'Validation_Score']
            self.write_log(model_folder_name + '!validation_log.csv', header)
            if write_info_message is not None and write:
                with open(model_folder_name + '!info.txt', 'w', newline='') as f:
                    f.write(write_info_message)

    def write_log(self, file_name, row_data):
        if self.write:
            with open(file_name, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)

    def load_model(self, file_name: str, model: PPO) -> int:
        """
        載入模型。
        :param file_name: 載入檔案的名稱。
        :param model: 所使用的演算法實例。
        :return: 當前的 episode 回數。
        """
        checkpoint = torch.load(self.model_folder_name + file_name)
        episode = checkpoint['episode']
        model.policy.load_state_dict(checkpoint['policy_state_dict'])
        model.old_policy.load_state_dict(checkpoint['policy_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return episode

    def save_model(self, episode: int, file_name: str, model: PPO):
        """
        儲存模型。
        :param episode: 當前的 episode 回數。
        :param file_name:  儲存檔案的名稱。
        :param model:  所使用的演算法實例。
        :return:
        """
        torch.save({
            'episode': episode,
            'policy_state_dict': model.policy.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict()
        }, self.model_folder_name + file_name)

    def training_log(self, episode: int, reward: float, loss: float, value_loss: float):
        """
        紀錄和打印訓練訊息。
        """
        row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), episode, reward, loss, value_loss]
        self.write_log(self.model_folder_name + '!training_log.csv', row)
        print('{}\t Episode {}\t Reward: {:.2f}\t Loss: {:.6f}\t Value_Loss: {:.6f}'.format(*row))

    def validation_log(self, episode: int, score: float):
        """
        紀錄和打印驗證訊息。
        """
        row = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), episode, score]
        self.write_log(self.model_folder_name + '!validation_log.csv', row)
        print('{}\t Validation_Score: {:.1f}'.format(row[0], row[2]))

