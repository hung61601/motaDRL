from collections import OrderedDict


class MatrixCOO:
    """使用 COO 格式儲存的稀疏鄰接矩陣類別。"""

    def __init__(self):
        self.indices: int = 0
        self.node_map: dict = {}
        self.node_links: OrderedDict = OrderedDict()
        self.col_indices: list = []  # x
        self.row_indices: list = []  # y

    def _node_id_to_indices(self) -> None:
        self.node_map.clear()
        self.indices = 0
        self.row_indices.clear()
        self.col_indices.clear()
        for node_id in self.node_links.keys():
            self.node_map[node_id] = self.indices
            self.indices += 1
        for start_node_id, end_node_ids in self.node_links.items():
            start_indices = self.node_map[start_node_id]
            for end_node_id in end_node_ids:
                if end_node_id in self.node_map:
                    end_indices = self.node_map[end_node_id]
                    self.col_indices.append(start_indices)
                    self.row_indices.append(end_indices)

    def get_info(self, update: bool = True) -> dict:
        if update:
            self._node_id_to_indices()
        return {'row': self.row_indices,
                'col': self.col_indices,
                'value': len(self.col_indices),
                'size': (self.indices, self.indices)}

    def add_node(self, node_id: tuple | str | int) -> None:
        self.node_links[node_id] = [node_id]

    def add_connect(self, start_node_id: tuple | str | int, end_node_id: tuple | str | int) -> None:
        self.node_links[start_node_id].append(end_node_id)

    def delete_node(self, node_id: tuple | str | int) -> None:
        self.node_links.pop(node_id)

    def delete_connect(self, start_node_id: tuple | str | int, end_node_id: tuple | str | int) -> None:
        if self.is_connected(start_node_id, end_node_id):
            self.node_links[start_node_id].remove(end_node_id)

    def is_connected(self, start_node_id: tuple | str | int, end_node_id: tuple | str | int) -> bool:
        if end_node_id in self.node_links[start_node_id]:
            return True
        return False

    def get_graph_node(self) -> iter:
        return self.node_links.keys()

    def get_indices(self, node_ids: list[tuple | str | int]) -> list[int]:
        return [self.node_map[node_id] for node_id in node_ids]

    def clear(self) -> None:
        self.indices = 0
        self.node_links.clear()
        self.row_indices.clear()
        self.col_indices.clear()
