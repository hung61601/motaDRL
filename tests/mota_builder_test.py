import pytest
from env.mota_builder import MotaBuilder


def test_build():
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
    player, events, events_map, template, player_id, end_id = builder.build()
    assert player.player_hp == 2700
    assert player.player_atk == 10
    assert player.player_def == 10
    assert player.yellow_key == 0
    assert player.blue_key == 0
    assert len(events) == 45
    assert events_map == {
        (0, 1): {(1, 1), (1, 2), (1, 3)},
        (1, 1): {(0, 1), (1, 2), (1, 3), (2, 1)},
        (1, 2): {(0, 1), (1, 1), (1, 3), (2, 2)},
        (1, 3): {(0, 1), (1, 1), (1, 2), (2, 3), (2, 4)},
        (2, 1): {(1, 1), (3, 1), (3, 2)},
        (2, 2): {(1, 2), (3, 3), (3, 4), (3, 5)},
        (2, 3): {(1, 3), (2, 4), (3, 6)},
        (2, 4): {(1, 3), (2, 3), (3, 7)},
        (3, 1): {(2, 1), (3, 2), (4, 1)},
        (3, 2): {(2, 1), (3, 1), (4, 2)},
        (3, 3): {(2, 2), (3, 4)},
        (3, 4): {(2, 2), (3, 3), (3, 5)},
        (3, 5): {(2, 2), (3, 4)},
        (3, 6): {(2, 3), (4, 3)},
        (3, 7): {(2, 4), (4, 4), (4, 5)},
        (4, 1): {(3, 1)},
        (4, 2): {(3, 2), (5, 1), (5, 2), (5, 3)},
        (4, 3): {(3, 6), (4, 4), (5, 4)},
        (4, 4): {(3, 7), (4, 3), (5, 4)},
        (4, 5): {(3, 7), (5, 5), (5, 6), (5, 7)},
        (5, 1): {(4, 2), (5, 2)},
        (5, 2): {(4, 2), (5, 1), (5, 3)},
        (5, 3): {(4, 2), (5, 2)},
        (5, 4): {(4, 3), (4, 4), (6, 1)},
        (5, 5): {(4, 5), (5, 6), (5, 7)},
        (5, 6): {(4, 5), (5, 5), (5, 7), (6, 2), (8, 1)},
        (5, 7): {(4, 5), (5, 5), (5, 6), (6, 3)},
        (6, 1): {(5, 4), (7, 1)},
        (6, 2): {(5, 6), (7, 2), (7, 3), (8, 1)},
        (6, 3): {(5, 7), (7, 4)},
        (7, 1): {(6, 1), (8, 1)},
        (7, 2): {(6, 2), (7, 3), (8, 2), (8, 3)},
        (7, 3): {(6, 2), (7, 2), (10, 1), (10, 2)},
        (7, 4): {(6, 3), (8, 5)},
        (8, 1): {(5, 6), (6, 2), (7, 1)},
        (8, 2): {(7, 2)},
        (8, 3): {(7, 2)},
        (8, 5): {(7, 4)},
        (11, 1): {(10, 1)},
        (10, 1): {(7, 3), (10, 2), (11, 1), (11, 3), (13, 1)},
        (10, 2): {(7, 3), (10, 1), (11, 3), (11, 4)},
        (11, 3): {(10, 1), (10, 2), (13, 1), (11, 4)},
        (11, 4): {(10, 2), (11, 3)},
        (14, 1): {(13, 1), (14, 2)},
        (13, 1): {(10, 1), (11, 3), (14, 1), (14, 2)},
        (14, 2): {(13, 1), (14, 1)}}
    assert len(template) == 14
    assert player_id == (0, 1)
    assert end_id == (14, 1)


def test_exception():
    builder = MotaBuilder()
    builder.reset()
    with pytest.raises(ValueError, match='The player has not been set.'):
        builder.add_enemy_template('yellowGuard', 30, 20, 17)
    with pytest.raises(ValueError, match='The player has not been set.'):
        builder.add_item_template('redPotion', 150, 0, 0, 0, 0)
    with pytest.raises(ValueError, match='The player or end event has not been set.'):
        builder.build()
    builder.set_player((0, 1), 2700, 10, 10, 0, 0)
    with pytest.raises(ValueError, match='The player or end event has not been set.'):
        builder.build()
