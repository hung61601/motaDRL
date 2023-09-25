import pytest
from env.matrix import MatrixCOO


def test_get_info():
    matrix = MatrixCOO()
    matrix.add_node('yellow')
    matrix.add_node('red_1')
    matrix.add_node('red_2')
    matrix.add_node('red_3')
    matrix.add_node('purple')
    matrix.add_node('blue_1')
    matrix.add_node('blue_2')
    matrix.add_node('blue_3')
    matrix.add_node('black')
    matrix.add_connect('yellow', 'red_1')
    matrix.add_connect('blue_1', 'yellow')
    matrix.add_connect('red_1', 'red_2')
    matrix.add_connect('red_2', 'red_1')
    matrix.add_connect('red_2', 'purple')
    matrix.add_connect('blue_1', 'blue_2')
    matrix.add_connect('blue_2', 'blue_1')
    matrix.add_connect('blue_2', 'purple')
    matrix.add_connect('purple', 'red_3')
    matrix.add_connect('purple', 'blue_3')
    matrix.add_connect('black', 'unknown')
    matrix.delete_node('red_2')
    assert matrix.get_info(update=False) == {'row': [], 'col': [], 'value': 0, 'size': (0, 0)}
    assert matrix.get_info(update=True) == {
        'col': [0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7],
        'row': [0, 1, 1, 2, 3, 2, 6, 4, 0, 5, 5, 4, 3, 6, 7],
        'size': (8, 8),
        'value': 15}


def test_add_node():
    matrix = MatrixCOO()
    matrix.add_node('root')
    assert 'root' in matrix.node_links


def test_add_connect():
    matrix = MatrixCOO()
    matrix.add_node('root')
    matrix.add_node('leaf')
    matrix.add_connect('root', 'leaf')
    assert 'leaf' in matrix.node_links['root']
    assert 'root' not in matrix.node_links['leaf']


def test_delete_node():
    matrix = MatrixCOO()
    matrix.add_node('root')
    matrix.delete_node('root')
    assert 'root' not in matrix.node_links


def test_delete_connect():
    matrix = MatrixCOO()
    matrix.add_node('root')
    matrix.add_node('leaf')
    matrix.add_connect('root', 'leaf')
    matrix.delete_connect('root', 'leaf')
    assert 'leaf' not in matrix.node_links['root']
    assert 'root' not in matrix.node_links['leaf']


def test_is_connected():
    matrix = MatrixCOO()
    matrix.add_node('root')
    matrix.add_node('leaf')
    matrix.add_connect('root', 'leaf')
    assert matrix.is_connected('root', 'leaf')
    assert not matrix.is_connected('leaf', 'root')
    matrix.delete_connect('root', 'leaf')
    assert not matrix.is_connected('root', 'leaf')
    assert not matrix.is_connected('leaf', 'root')


def test_get_graph_node():
    matrix = MatrixCOO()
    matrix.add_node('yellow')
    matrix.add_node('red_1')
    matrix.add_node('red_2')
    matrix.add_node('red_3')
    matrix.add_node('purple')
    matrix.add_node('blue_1')
    matrix.add_node('blue_2')
    matrix.add_node('blue_3')
    matrix.add_node('black')
    matrix.add_connect('yellow', 'red_1')
    matrix.add_connect('blue_1', 'yellow')
    matrix.add_connect('red_1', 'red_2')
    matrix.add_connect('red_2', 'red_1')
    matrix.add_connect('red_2', 'purple')
    matrix.add_connect('blue_1', 'blue_2')
    matrix.add_connect('blue_2', 'blue_1')
    matrix.add_connect('blue_2', 'purple')
    matrix.add_connect('purple', 'red_3')
    matrix.add_connect('purple', 'blue_3')
    assert list(matrix.get_graph_node()) == [
        'yellow', 'red_1', 'red_2', 'red_3', 'purple', 'blue_1', 'blue_2', 'blue_3', 'black']
    matrix.delete_node('red_2')
    assert list(matrix.get_graph_node()) == [
        'yellow', 'red_1', 'red_3', 'purple', 'blue_1', 'blue_2', 'blue_3', 'black']


def test_get_indices():
    matrix = MatrixCOO()
    matrix.add_node('yellow')
    matrix.add_node('red_1')
    matrix.add_node('red_2')
    matrix.add_node('red_3')
    matrix.add_node('purple')
    matrix.add_node('blue_1')
    matrix.add_node('blue_2')
    matrix.add_node('blue_3')
    matrix.delete_node('red_2')
    matrix.add_node('black')
    with pytest.raises(KeyError):
        matrix.get_indices(['yellow', 'purple', 'black', 'red_1', 'red_3'])
    matrix._node_id_to_indices()
    assert matrix.get_indices(['yellow', 'purple', 'black', 'red_1', 'red_3']) == [0, 3, 7, 1, 2]


def test_clear():
    matrix = MatrixCOO()
    matrix.add_node('yellow')
    matrix.add_node('red_1')
    matrix.add_node('red_2')
    matrix.add_node('red_3')
    matrix.add_node('purple')
    matrix.add_node('blue_1')
    matrix.add_node('blue_2')
    matrix.add_node('blue_3')
    matrix.add_node('black')
    matrix.add_connect('yellow', 'red_1')
    matrix.add_connect('blue_1', 'yellow')
    matrix.add_connect('red_1', 'red_2')
    matrix.add_connect('red_2', 'red_1')
    matrix.add_connect('red_2', 'purple')
    matrix.add_connect('blue_1', 'blue_2')
    matrix.add_connect('blue_2', 'blue_1')
    matrix.add_connect('blue_2', 'purple')
    matrix.add_connect('purple', 'red_3')
    matrix.add_connect('purple', 'blue_3')
    matrix._node_id_to_indices()
    assert matrix.indices == 9
    assert len(matrix.node_links) > 0
    assert len(matrix.row_indices) > 0
    assert len(matrix.col_indices) > 0
    matrix.clear()
    assert matrix.indices == 0
    assert len(matrix.node_links) == 0
    assert len(matrix.row_indices) == 0
    assert len(matrix.col_indices) == 0
