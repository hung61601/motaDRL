import gymnasium as gym
from env.matrix import MatrixCOO
from env.event import Player, Event


class Mota(gym.Env, gym.utils.EzPickle):

    def __init__(self,
                 graphic_depth: int = 5,
                 use_advanced_feature: bool = True):
        gym.utils.EzPickle.__init__(self)
        self.graphic_depth: int = graphic_depth
        self.use_advanced_feature: bool = use_advanced_feature
        self.matrix: MatrixCOO = MatrixCOO()
        self.candidate_events: list[tuple | str | int] = []
        self.visited_events: dict[tuple | str | int, int] = {}
        self.activated_events: set[tuple | str | int] = set()
        self.score = 0
        self.player: Player | None = None
        self.events: dict[tuple | str | int, Event] = {}
        self.events_map: dict[tuple | str | int, set] = {}
        self.events_template: dict[tuple | str | int, Event] = {}
        self.player_id: tuple | str | int = 0
        self.end_id: tuple | str | int = 0

    def _reset_candidate(self) -> None:
        """
        重新設定候選事件。
        """
        self.candidate_events = list(self.events_map[self.player_id])

    def _remove_player_event(self) -> None:
        """
        從地圖中移除其他事件往主角事件的連結。
        """
        for event_id in self.events_map[self.player_id]:
            self.events_map[event_id].remove(self.player_id)

    def _reset_matrix(self) -> None:
        """
        重新建立鄰接矩陣，不包含主角節點。
        """
        self.matrix.clear()
        self.visited_events.clear()
        self.activated_events.clear()
        self.extend_graph(self.player_id)

    def update_events_feature(self) -> None:
        """
        僅更新圖形中的事件，減少更新多事件造成效能下降。
        """
        unique_events = set(self.events[event_id] for event_id in self.matrix.get_graph_node())
        for event in unique_events:
            event.update_feature()

    def extend_graph(self, event_id) -> None:
        """
        將地圖最大深度內的事件加進圖形中。
        :param event_id: 事件編號。
        """
        queue = [(event_id, 0)]
        while queue:
            top_event, depth = queue.pop(0)
            depth += 1
            if depth > self.graphic_depth:
                continue
            for event_id in self.events_map[top_event]:
                # 出現新節點時，在圖形中建立其節點和連線。
                if event_id not in self.visited_events:
                    self.matrix.add_node(event_id)
                    for adjacent_event in self.events_map[event_id]:
                        self.matrix.add_connect(event_id, adjacent_event)
                    self.visited_events[event_id] = depth
                    queue.append((event_id, depth))
                # 如果現在深度比先前走訪該節點的深度還要低，則需要進行走訪以獲取更深度的節點。
                elif depth < self.visited_events[event_id]:
                    self.visited_events[event_id] = depth
                    queue.append((event_id, depth))

    def update_candidate_events(self, event_id) -> None:
        """
        更新候選事件。
        :param event_id: 事件編號。
        """
        self.candidate_events.remove(event_id)
        for adjacent_event in self.events_map[event_id]:
            if adjacent_event not in self.activated_events and adjacent_event not in self.candidate_events:
                self.candidate_events.append(adjacent_event)

    def get_feature(self) -> dict:
        """
        獲取圖形特徵。
        :returns:
            node: 節點特徵。
            graph: 全體特徵。
        """
        node_features = []
        for event_id in self.matrix.get_graph_node():
            node_features.append(self.events[event_id].get_feature(self.use_advanced_feature))
        return {'node': node_features,
                'graph': self.player.get_feature()}

    def get_info(self) -> dict:
        """
        獲取環境訊息。
        :returns:
            adj_matrix: 鄰接矩陣。
            candidate: 候選事件索引。
            mask: 不可激活的候選事件遮罩。
        """
        return {'adj_matrix': self.matrix.get_info(),
                'candidate': self.matrix.get_indices(self.candidate_events),
                'mask': [not self.events[event_id].can_activated() for event_id in self.candidate_events]}

    def step(self, action: int):
        selected_event_id = self.candidate_events[action]
        # 激活事件
        selected_event = self.events[selected_event_id]
        selected_event.activate()
        self.activated_events.add(selected_event_id)
        # 更新相鄰矩陣
        self.extend_graph(selected_event_id)
        self.matrix.delete_node(selected_event_id)
        # 更新候選事件
        self.update_candidate_events(selected_event_id)
        # 計算獎勵
        reward = self.player.player_hp - self.score
        self.score = self.player.player_hp
        # 更新特徵
        self.update_events_feature()
        # 判斷是否結束
        info = self.get_info()
        done = (selected_event_id == self.end_id) or all(info['mask'])
        return self.get_feature(), reward, done, False, info

    def reset(self,
              player: Player = None,
              events: dict[tuple | str | int, Event] = None,
              events_map: dict[tuple | str | int, set] = None,
              events_template: dict[tuple | str | int, Event] = None,
              player_id: tuple | str | int = None,
              end_id: tuple | str | int = None) -> tuple[dict, dict]:
        self.player = player
        self.events = events
        self.events_map = events_map
        self.events_template = events_template
        self.player_id = player_id
        self.end_id = end_id
        self._reset_candidate()
        self._remove_player_event()
        self._reset_matrix()
        self.update_events_feature()
        self.score = self.player.player_hp
        return self.get_feature(), self.get_info()
