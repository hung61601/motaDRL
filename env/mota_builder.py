from env.event import Player, Event, Enemy, Item


class MotaBuilder:

    def __init__(self):
        self.player: Player | None = None
        self.events: dict[tuple | str | int, Event] = {}
        self.template: dict[tuple | str | int, Event] = {}
        self.events_map: dict[tuple | str | int, set] = {}
        self.player_id: tuple | str | int | None = None
        self.end_id: tuple | str | int | None = None

    def reset(self):
        """
        清空建立魔塔資料所需的設定。
        """
        self.player = None
        self.events = {}
        self.template = {}
        self.events_map = {}
        self.player_id = None
        self.end_id = None

    def set_player(self,
                   event_id: tuple | str | int,
                   player_hp: int,
                   player_atk: int,
                   player_def: int,
                   yellow_key: int,
                   blue_key: int) -> None:
        """
        設定主角資料。
        :param event_id: 能識別事件的唯一編號，例如事件的座標。
        :param player_hp: 主角的生命值。
        :param player_atk: 主角的攻擊力。
        :param player_def: 主角的防禦力。
        :param yellow_key: 主角持有的黃鑰匙數量。
        :param blue_key: 主角持有的藍鑰匙數量。
        """
        self.player_id = event_id
        self.player = Player(player_hp, player_atk, player_def, yellow_key, blue_key)

    def set_end(self,
                event_id: tuple | str | int) -> None:
        """
        設定終點事件。
        :param event_id: 能識別事件的唯一編號，例如事件的座標。
        """
        self.end_id = event_id
        self.events[event_id] = Item(self.player, 0, 0, 0, 0, 0)

    def add_enemy_template(self,
                           template_id: tuple | str | int,
                           enemy_hp: int,
                           enemy_atk: int,
                           enemy_def: int) -> None:
        """
        添加敵人事件模板。
        :param template_id: 能識別事件模板的唯一編號。
        :param enemy_hp: 敵人的生命值。
        :param enemy_atk: 敵人的攻擊力。
        :param enemy_def: 敵人的防禦力。
        """
        if self.player is None:
            raise ValueError('The player has not been set.')
        self.template[template_id] = Enemy(self.player, enemy_hp, enemy_atk, enemy_def)

    def add_item_template(self,
                          template_id: tuple | str | int,
                          inc_hp: int,
                          inc_atk: int,
                          inc_def: int,
                          yellow_key: int,
                          blue_key: int) -> None:
        """
        添加道具事件模板。
        :param template_id: 能識別事件模板的唯一編號。
        :param inc_hp: 獲取道具時，主角增加的生命量。
        :param inc_atk: 獲取道具時，主角增加的攻擊力。
        :param inc_def: 獲取道具時，主角增加的防禦力。
        :param yellow_key: 獲取道具時，主角增加的黃鑰匙數量。
        :param blue_key: 獲取道具時，主角增加的藍鑰匙數量。
        """
        if self.player is None:
            raise ValueError('The player has not been set.')
        self.template[template_id] = Item(self.player, inc_hp, inc_atk, inc_def, yellow_key, blue_key)

    def add_event(self,
                  event_id: tuple | str | int,
                  template_id: tuple | str | int,) -> None:
        """
        添加事件。
        :param event_id: 能識別事件的唯一編號，例如事件的座標。
        :param template_id: 事件模板的編號。
        """
        self.events[event_id] = self.template[template_id]

    def add_links(self,
                  *args: tuple | str | int) -> None:
        """
        添加事件之間連接的道路。
        :param args: 要連接的事件標號，事件之間進行全連接。
        """
        for event_a in args:
            if event_a not in self.events_map:
                self.events_map[event_a] = set()
            self.events_map[event_a].update(args)
            self.events_map[event_a].remove(event_a)

    def build(self) -> tuple[
            Player,
            dict[tuple | str | int, Event],
            dict[tuple | str | int, set],
            dict[tuple | str | int, Event],
            tuple | str | int,
            tuple | str | int]:
        """
        建立環境。
        :return: 建立好魔塔環境所需資料。
        """
        if self.player_id is None or self.end_id is None:
            raise ValueError('The player or end event has not been set.')
        return self.player, self.events, self.events_map, self.template, self.player_id, self.end_id
