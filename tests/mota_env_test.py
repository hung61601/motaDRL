import copy
from env.mota_env import Mota
from env.mota_builder import MotaBuilder


def test_update_graph_feature():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player(-1, 100, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_event(0, 'slime')
    builder.add_event(1, 'potion')
    builder.set_end(2)
    builder.add_links(-1, 0, 1)
    builder.add_links(1, 2)
    player, events, events_map, template, player_id, end_id = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    env.reset(player, events, events_map, template, player_id, end_id)
    events[0].enemy_atk = 20
    player.player_hp += 200
    assert env.get_feature() == {
        'node': [
            (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (300, 10, 10, 0, 0)
    }
    env.update_events_feature()
    assert env.get_feature() == {
        'node': [
            (-40, 0, 0, 0, 0, 2.5, 10, 4, 0),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (300, 10, 10, 0, 0)
    }


def test_extend_graph():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player(-1, 100, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_event(0, 'slime')
    builder.add_event(1, 'potion')
    builder.add_event(2, 'slime')
    builder.add_event(3, 'potion')
    builder.set_end(4)
    builder.add_links(-1, 0)
    builder.add_links(0, 1)
    builder.add_links(1, 2)
    builder.add_links(1, 3)
    builder.add_links(3, 4)
    mota_data = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    env.reset(*mota_data)
    assert env.get_info() == {
        'adj_matrix': {
            'row': [0, 1, 1, 0],
            'col': [0, 0, 1, 1],
            'value': 4,
            'size': (2, 2)
        },
        'candidate': [0],
        'mask': [False]
    }
    env.extend_graph(0)
    assert env.get_info() == {
        'adj_matrix': {
            'row': [0, 1, 1, 0, 2, 3, 2, 1, 3, 1],
            'col': [0, 0, 1, 1, 1, 1, 2, 2, 3, 3],
            'value': 10,
            'size': (4, 4)
        },
        'candidate': [0],
        'mask': [False]
    }


def test_update_candidate_events():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player(-1, 100, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_event(0, 'slime')
    builder.add_event(1, 'potion')
    builder.add_event(2, 'slime')
    builder.set_end(3)
    builder.add_links(-1, 0, 1)
    builder.add_links(1, 2)
    builder.add_links(0, 3)
    builder.add_links(2, 3)
    mota_data = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    env.reset(*mota_data)
    assert env.candidate_events == [0, 1]
    env.activated_events.add(1)
    env.update_candidate_events(1)
    assert env.candidate_events == [0, 2]
    env.activated_events.add(2)
    env.update_candidate_events(2)
    assert env.candidate_events == [0, 3]


def test_get_feature():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player(-1, 100, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_event(0, 'slime')
    builder.add_event(1, 'potion')
    builder.set_end(2)
    builder.add_links(-1, 0, 1)
    builder.add_links(1, 2)
    mota_data = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    env.reset(*mota_data)
    assert env.get_feature() == {
        'node': [
            (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (100, 10, 10, 0, 0)
    }


def test_get_info():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player(-1, 100, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_event(0, 'slime')
    builder.add_event(1, 'potion')
    builder.set_end(2)
    builder.add_links(-1, 0, 1)
    builder.add_links(1, 2)
    mota_data = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    env.reset(*mota_data)
    assert env.get_info() == {
        'adj_matrix': {
            'row': [0, 1, 1, 0, 2, 2, 1],
            'col': [0, 0, 1, 1, 1, 2, 2],
            'value': 7,
            'size': (3, 3)
        },
        'candidate': [0, 1],
        'mask': [False, False]
    }


def test_step():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player((0, 1), 500, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_enemy_template('bat', 30, 15, 4)
    builder.add_enemy_template('skeleton', 60, 18, 6)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_item_template('jewel', 0, 2, 0, 0, 0)
    builder.add_item_template('key', 0, 0, 0, 1, 0)
    builder.add_item_template('door', 0, 0, 0, -1, 0)
    builder.add_event((1, 1), 'slime')
    builder.add_event((1, 2), 'key')
    builder.add_event((2, 1), 'bat')
    builder.add_event((2, 2), 'door')
    builder.add_event((3, 1), 'potion')
    builder.add_event((3, 2), 'jewel')
    builder.add_event((3, 3), 'jewel')
    builder.add_event((4, 1), 'slime')
    builder.add_event((5, 1), 'skeleton')
    builder.add_event((5, 2), 'potion')
    builder.set_end((6, 1))
    builder.add_links((0, 1), (1, 1))
    builder.add_links((0, 1), (1, 2))
    builder.add_links((1, 1), (2, 1))
    builder.add_links((1, 2), (2, 2))
    builder.add_links((1, 2), (4, 1))
    builder.add_links((2, 1), (3, 1), (3, 2))
    builder.add_links((2, 2), (3, 3))
    builder.add_links((3, 2), (4, 1))
    builder.add_links((4, 1), (5, 1), (5, 2))
    builder.add_links((5, 1), (6, 1))
    mota_data = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    env.reset(*mota_data)
    features, reward, done, _, info = env.step(env.candidate_events.index((1, 1)))
    assert features == {
        'node': [
            (0, 0, 0, 1, 0, 0, 0, 0, 0),
            (-20, 0, 0, 0, 0, 1.5, 5, 4, 1),
            (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
            (0, 0, 0, -1, 0, 0, 0, 0, 0),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (492, 10, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 2, 3, 1, 4, 5, 2, 0, 5, 3, 0, 4, 5, 1, 5, 4, 2, 1],
            'col': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5],
            'value': 18,
            'size': (6, 6)
        },
        'candidate': [0, 1],
        'mask': [False, False]
    }
    assert reward == -8
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((2, 1)))
    assert features == {
        'node': [
            (0, 0, 0, 1, 0, 0, 0, 0, 0),
            (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
            (0, 0, 0, -1, 0, 0, 0, 0, 0),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (472, 10, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 1, 2, 1, 0, 4, 2, 0, 3, 4, 4, 3, 1],
            'col': [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
            'value': 13,
            'size': (5, 5)
        },
        'candidate': [0, 3, 4],
        'mask': [False, False, False]
    }
    assert reward == -20
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((1, 2)))
    assert features == {
        'node': [
            (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
            (0, 0, 0, -1, 0, 0, 0, 0, 0),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0),
            (-112, 0, 0, 0, 0, 0.2857142857142856, 8, 14, -4),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (472, 10, 10, 1, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 4, 3, 5, 1, 6, 2, 3, 3, 2, 0, 4, 0, 5, 5, 0, 4, 6, 1],
            'col': [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6],
            'value': 19,
            'size': (7, 7)
        },
        'candidate': [2, 3, 0, 1],
        'mask': [False, False, False, False]
    }
    assert reward == 0
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((2, 2)))
    assert features == {
        'node': [
            (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0),
            (-112, 0, 0, 0, 0, 0.2857142857142856, 8, 14, -4),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (472, 10, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 3, 2, 4, 1, 2, 2, 1, 0, 3, 0, 4, 4, 0, 3, 5],
            'col': [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
            'value': 16,
            'size': (6, 6)
        },
        'candidate': [1, 2, 0, 5],
        'mask': [False, False, False, False]
    }
    assert reward == 0
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((3, 3)))
    assert features == {
        'node': [
            (-8, 0, 0, 0, 0, 0.5, 2, 4, 10),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0),
            (-72, 0, 0, 0, 0, 0.666666666666667, 8, 9, -2),
            (100, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (472, 12, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 3, 2, 4, 1, 2, 2, 1, 0, 3, 0, 4, 4, 0, 3],
            'col': [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'value': 15,
            'size': (5, 5)
        },
        'candidate': [1, 2, 0],
        'mask': [False, False, False]
    }
    assert reward == 0
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((4, 1)))
    assert features == {
        'node': [
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0, 0, 0, 0),
            (-72, 0, 0, 0, 0, 0.666666666666667, 8, 9, -2),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (464, 12, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 1, 1, 0, 2, 4, 3, 3, 2, 4, 2],
            'col': [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            'value': 11,
            'size': (5, 5)
        },
        'candidate': [0, 1, 2, 3],
        'mask': [False, False, False, False]
    }
    assert reward == -8
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((3, 2)))
    assert features == {
        'node': [
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (-56, 0, 0, 0, 0, 0.5714285714285712, 8, 7, 0),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (464, 14, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 1, 3, 2, 2, 1, 3, 1],
            'col': [0, 1, 1, 1, 2, 2, 3, 3],
            'value': 8,
            'size': (4, 4)
        },
        'candidate': [0, 1, 2],
        'mask': [False, False, False]
    }
    assert reward == 0
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((3, 1)))
    assert features == {
        'node': [
            (-56, 0, 0, 0, 0, 0.5714285714285712, 8, 7, 0),
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (564, 14, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 2, 1, 1, 0, 2, 0],
            'col': [0, 0, 0, 1, 1, 2, 2],
            'value': 7,
            'size': (3, 3)
        },
        'candidate': [0, 1],
        'mask': [False, False]
    }
    assert reward == 100
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((5, 1)))
    assert features == {
        'node': [
            (100, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0)
        ],
        'graph': (508, 14, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0, 1],
            'col': [0, 1],
            'value': 2,
            'size': (2, 2)
        },
        'candidate': [0, 1],
        'mask': [False, False]
    }
    assert reward == -56
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((5, 2)))
    assert features == {
        'node': [(0, 0, 0, 0, 0, 0, 0, 0, 0)],
        'graph': (608, 14, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [0],
            'col': [0],
            'value': 1,
            'size': (1, 1)
        },
        'candidate': [0],
        'mask': [False]
    }
    assert reward == 100
    assert not done
    features, reward, done, _, info = env.step(env.candidate_events.index((6, 1)))
    assert features == {
        'node': [],
        'graph': (608, 14, 10, 0, 0)
    }
    assert info == {
        'adj_matrix': {
            'row': [],
            'col': [],
            'value': 0,
            'size': (0, 0)
        },
        'candidate': [],
        'mask': []
    }
    assert reward == 0
    assert done


def test_reset():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player((0, 1), 500, 10, 10, 0, 0)
    builder.add_enemy_template('slime', 50, 12, 0)
    builder.add_enemy_template('bat', 30, 15, 4)
    builder.add_enemy_template('skeleton', 60, 18, 6)
    builder.add_item_template('potion', 100, 0, 0, 0, 0)
    builder.add_item_template('jewel', 0, 2, 0, 0, 0)
    builder.add_item_template('key', 0, 0, 0, 1, 0)
    builder.add_item_template('door', 0, 0, 0, -1, 0)
    builder.add_event((1, 1), 'slime')
    builder.add_event((1, 2), 'key')
    builder.add_event((2, 1), 'bat')
    builder.add_event((2, 2), 'door')
    builder.add_event((3, 1), 'potion')
    builder.add_event((3, 2), 'jewel')
    builder.add_event((3, 3), 'jewel')
    builder.add_event((4, 1), 'slime')
    builder.add_event((5, 1), 'skeleton')
    builder.add_event((5, 2), 'potion')
    builder.set_end((6, 1))
    builder.add_links((0, 1), (1, 1))
    builder.add_links((0, 1), (1, 2))
    builder.add_links((1, 1), (2, 1))
    builder.add_links((1, 2), (2, 2))
    builder.add_links((1, 2), (4, 1))
    builder.add_links((2, 1), (3, 1), (3, 2))
    builder.add_links((2, 2), (3, 3))
    builder.add_links((3, 2), (4, 1))
    builder.add_links((4, 1), (5, 1), (5, 2))
    builder.add_links((5, 1), (6, 1))
    mota_data = builder.build()
    env = Mota(graphic_depth=2, use_advanced_feature=True)
    for _ in range(2):
        features, info = env.reset(*copy.deepcopy(mota_data))
        assert features == {
            'node': [
                (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
                (0, 0, 0, 1, 0, 0, 0, 0, 0),
                (-20, 0, 0, 0, 0, 1.5, 5, 4, 1),
                (-8, 0, 0, 0, 0, 2.5, 2, 4, 8),
                (0, 0, 0, -1, 0, 0, 0, 0, 0)
            ],
            'graph': (500, 10, 10, 0, 0)
        }
        assert info == {
            'adj_matrix': {
                'row': [0, 2, 1, 3, 4, 2, 0, 3, 1, 4, 1],
                'col': [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                'value': 11,
                'size': (5, 5)
            },
            'candidate': [0, 1],
            'mask': [False, False]
        }


def test_mota_ai_map02_simulation():
    builder = MotaBuilder()
    builder.reset()
    builder.set_player((0, 1), 2700, 10, 10, 0, 0)
    builder.add_enemy_template('greenSlime', 50, 14, 5)
    builder.add_enemy_template('redSlime', 80, 13, 6)
    builder.add_enemy_template('bat', 30, 20, 4)
    builder.add_enemy_template('skeleton', 60, 18, 10)
    builder.add_enemy_template('bluePriest', 100, 22, 15)
    builder.add_enemy_template('blackSlime', 120, 25, 13)
    builder.add_enemy_template('yellowGuard', 30, 20, 17)
    builder.add_item_template('redPotion', 150, 0, 0, 0, 0)
    builder.add_item_template('redJewel', 0, 2, 0, 0, 0)
    builder.add_item_template('blueJewel', 0, 0, 2, 0, 0)
    builder.add_item_template('yellowKey', 0, 0, 0, 1, 0)
    builder.add_item_template('blueKey', 0, 0, 0, 0, 1)
    builder.add_item_template('yellowDoor', 0, 0, 0, -1, 0)
    builder.add_item_template('blueDoor', 0, 0, 0, 0, -1)
    builder.add_event((1, 1), 'greenSlime')
    builder.add_event((1, 2), 'yellowDoor')
    builder.add_event((1, 3), 'greenSlime')
    builder.add_event((2, 1), 'yellowKey')
    builder.add_event((2, 2), 'redSlime')
    builder.add_event((2, 3), 'redJewel')
    builder.add_event((2, 4), 'redPotion')
    builder.add_event((3, 1), 'bat')
    builder.add_event((3, 2), 'yellowDoor')
    builder.add_event((3, 3), 'redPotion')
    builder.add_event((3, 4), 'redJewel')
    builder.add_event((3, 5), 'redPotion')
    builder.add_event((3, 6), 'redSlime')
    builder.add_event((3, 7), 'greenSlime')
    builder.add_event((4, 1), 'blueJewel')
    builder.add_event((4, 2), 'skeleton')
    builder.add_event((4, 3), 'yellowKey')
    builder.add_event((4, 4), 'bat')
    builder.add_event((4, 5), 'yellowDoor')
    builder.add_event((5, 1), 'yellowKey')
    builder.add_event((5, 2), 'redPotion')
    builder.add_event((5, 3), 'blueJewel')
    builder.add_event((5, 4), 'bluePriest')
    builder.add_event((5, 5), 'blueJewel')
    builder.add_event((5, 6), 'blackSlime')
    builder.add_event((5, 7), 'skeleton')
    builder.add_event((6, 1), 'redPotion')
    builder.add_event((6, 2), 'blueDoor')
    builder.add_event((6, 3), 'redJewel')
    builder.add_event((7, 1), 'redJewel')
    builder.add_event((7, 2), 'blueJewel')
    builder.add_event((7, 3), 'yellowGuard')
    builder.add_event((7, 4), 'redPotion')
    builder.add_event((8, 1), 'bluePriest')
    builder.add_event((8, 2), 'redPotion')
    builder.add_event((8, 3), 'redPotion')
    builder.add_event((8, 5), 'blueKey')
    builder.add_event((10, 1), 'redSlime')
    builder.add_event((10, 2), 'redPotion')
    builder.add_event((11, 1), 'yellowKey')
    builder.add_event((11, 3), 'yellowDoor')
    builder.add_event((11, 4), 'yellowKey')
    builder.add_event((13, 1), 'yellowDoor')
    builder.set_end((14, 1))
    builder.add_event((14, 2), 'redJewel')
    builder.add_links((0, 1), (1, 1), (1, 2), (1, 3))
    builder.add_links((1, 1), (2, 1))
    builder.add_links((1, 2), (2, 2))
    builder.add_links((1, 3), (2, 3), (2, 4))
    builder.add_links((2, 1), (3, 1), (3, 2))
    builder.add_links((2, 2), (3, 3), (3, 4))
    builder.add_links((2, 2), (3, 4), (3, 5))
    builder.add_links((2, 3), (3, 6))
    builder.add_links((2, 4), (3, 7))
    builder.add_links((3, 1), (4, 1))
    builder.add_links((3, 2), (4, 2))
    builder.add_links((3, 6), (4, 3))
    builder.add_links((3, 7), (4, 4))
    builder.add_links((3, 7), (4, 5))
    builder.add_links((4, 2), (5, 1), (5, 2))
    builder.add_links((4, 2), (5, 2), (5, 3))
    builder.add_links((4, 3), (4, 4), (5, 4))
    builder.add_links((4, 5), (5, 5), (5, 6), (5, 7))
    builder.add_links((5, 4), (6, 1))
    builder.add_links((5, 6), (6, 2), (8, 1))
    builder.add_links((5, 7), (6, 3))
    builder.add_links((6, 1), (7, 1))
    builder.add_links((6, 2), (7, 2), (7, 3))
    builder.add_links((6, 3), (7, 4))
    builder.add_links((7, 1), (8, 1))
    builder.add_links((7, 2), (8, 2))
    builder.add_links((7, 2), (8, 3))
    builder.add_links((7, 3), (10, 1), (10, 2))
    builder.add_links((7, 4), (8, 5))
    builder.add_links((10, 1), (11, 1))
    builder.add_links((10, 1), (11, 3), (13, 1))
    builder.add_links((10, 2), (11, 3), (11, 4))
    builder.add_links((13, 1), (14, 1), (14, 2))
    mota_data = builder.build()
    env = Mota(graphic_depth=5, use_advanced_feature=True)
    _, info = env.reset(*mota_data)
    actions_list = [
        (1, 1), (2, 1), (3, 1), (4, 1), (1, 3), (2, 3), (2, 4), (1, 2), (2, 2), (3, 4), (3, 3), (3, 5),
        (3, 6), (4, 3), (3, 2), (4, 2), (5, 2), (5, 3), (5, 1), (3, 7), (4, 5), (5, 5), (5, 7), (6, 3),
        (7, 4), (8, 5), (5, 6), (6, 2), (7, 2), (8, 2), (8, 3), (8, 1), (7, 1), (6, 1), (7, 3), (10, 2),
        (11, 4), (10, 1), (11, 1), (13, 1), (14, 2), (14, 1)]
    expected_rewards = [
        -36, 0, -40, 0, -18, 0, 150, 0, -13, 0, 150, 150, -9, 0, 0, -84, 150, 0, 0, 0, 0, 0, -28, 0,
        150, 0, -351, 0, 0, 150, 150, -396, 0, 150, -58, 150, 0, 0, 0, 0, 0, 0]
    expected_dones = [False] * 41 + [True]
    for i in range(len(actions_list)):
        action = env.candidate_events.index(actions_list[i])
        assert not info['mask'][action]
        _, reward, done, _, info = env.step(action)
        assert reward == expected_rewards[i]
        assert done == expected_dones[i]
    assert env.player.player_hp == 3017
