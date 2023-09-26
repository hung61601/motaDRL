from env.util import condition_clamp


class Player:
    """主角類別。"""

    def __init__(self,
                 player_hp: int,
                 player_atk: int,
                 player_def: int,
                 yellow_key: int,
                 blue_key: int):
        self.player_hp = player_hp
        self.player_atk = player_atk
        self.player_def = player_def
        self.yellow_key = yellow_key
        self.blue_key = blue_key

    def get_feature(self):
        return self.player_hp, self.player_atk, self.player_def, self.yellow_key, self.blue_key


class Event:
    """所有事件的父類別。"""

    def __init__(self,
                 player: Player):
        self.player = player
        self.enemy_hp: int = 0          # 敵人生命值
        self.enemy_atk: int = 0         # 敵人攻擊力
        self.enemy_def: int = 0         # 敵人防禦力
        self.inc_hp: int = 0            # 主角增加生命
        self.inc_atk: int = 0           # 主角增加攻擊
        self.inc_def: int = 0           # 主角增加防禦
        self.yellow_key: int = 0        # 主角增加黃鑰匙
        self.blue_key: int = 0          # 主角增加藍鑰匙
        self.threshold_demand: int = 0  # 攻擊臨界值(還差增加多少攻擊能減少一次敵人戰鬥攻擊回數)
        self.atk_profit: float = 0      # 戰鬥減少一回時能省下的傷害
        self.def_profit: float = 0      # 每增加一點防禦能省下的傷害
        self.balance: int = 0           # 攻防平衡值

    def activate(self):
        """
        激活事件。會影響主角的狀態。
        """
        self.player.player_hp += self.inc_hp
        self.player.player_atk += self.inc_atk
        self.player.player_def += self.inc_def
        self.player.yellow_key += self.yellow_key
        self.player.blue_key += self.blue_key

    def can_activated(self) -> bool:
        """
        是否可以激活事件。
        """
        return True

    def update_feature(self) -> None:
        """
        更新事件特徵，以改變激活事件時影響的狀態。
        """
        pass

    def get_feature(self, is_advanced: bool = False) -> tuple:
        """
        獲取事件特徵。
        :param is_advanced: 默認 False。當為 True 時，替換事件部分特徵。
        :return: 事件的特徵。
        """
        if is_advanced:
            return (self.inc_hp, self.inc_atk, self.inc_def, self.yellow_key, self.blue_key,
                    self.threshold_demand, self.atk_profit, self.def_profit, self.balance)
        else:
            return (self.inc_hp, self.inc_atk, self.inc_def, self.yellow_key, self.blue_key,
                    self.enemy_hp, self.enemy_atk, self.enemy_def)


class Enemy(Event):
    """怪物事件類別。"""

    def __init__(self,
                 player: Player,
                 enemy_hp,
                 enemy_atk,
                 enemy_def):
        super().__init__(player)
        self.enemy_hp = enemy_hp
        self.enemy_atk = enemy_atk
        self.enemy_def = enemy_def

    def _battle(self) -> None:
        """
        計算戰鬥結果。
        """
        p_damage = self.player.player_atk - self.enemy_def
        e_damage = self.enemy_atk - self.player.player_def
        if p_damage <= 0:
            rounds = 0
            damage = self.player.player_hp
            self.threshold_demand = -p_damage + 1
        else:
            rounds = self.enemy_hp // p_damage - (self.enemy_hp % p_damage == 0)
            damage = e_damage * rounds
            damage = condition_clamp(damage, self.player.player_hp, 0)
            self.threshold_demand = self.enemy_hp / rounds - p_damage
        self.atk_profit = e_damage  # 當主角傷害 = 0 時，攻省為無限大，但此處取怪物單回合傷害作為上限
        self.def_profit = rounds
        self.balance = p_damage - e_damage
        self.inc_hp = -damage

    def can_activated(self):
        return -self.inc_hp < self.player.player_hp

    def update_feature(self) -> None:
        self._battle()


class Item(Event):
    """道具事件類別。"""

    def __init__(self,
                 player: Player,
                 inc_hp: int,
                 inc_atk: int,
                 inc_def: int,
                 yellow_key: int,
                 blue_key: int
                 ):
        super().__init__(player)
        self.inc_hp = inc_hp
        self.inc_atk = inc_atk
        self.inc_def = inc_def
        self.yellow_key = yellow_key
        self.blue_key = blue_key

    def can_activated(self):
        if -self.yellow_key > self.player.yellow_key:
            return False
        elif -self.blue_key > self.player.blue_key:
            return False
        else:
            return True
