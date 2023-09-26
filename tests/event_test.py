from env.event import Player, Event, Enemy, Item


def test_player():
    player = Player(500, 15, 12, 1, 2)
    assert player.get_feature() == (500, 15, 12, 1, 2)


def test_event_subclass():
    assert issubclass(Enemy, Event)
    assert issubclass(Item, Event)


def test_event_activate():
    player = Player(1000, 10, 10, 0, 0)
    event = Event(player)
    assert event.can_activated()


def test_enemy_activate():
    player = Player(1000, 10, 10, 0, 0)
    enemy = Enemy(player, 50, 18, 6)
    enemy.update_feature()
    enemy.activate()
    assert player.player_hp == 904


def test_enemy_can_activated():
    player = Player(100, 10, 10, 0, 0)
    weak_enemy = Enemy(player, 50, 18, 6)
    weak_enemy.update_feature()
    assert weak_enemy.can_activated()
    powerful_enemy = Enemy(player, 50, 80, 6)
    powerful_enemy.update_feature()
    assert not powerful_enemy.can_activated()
    invincible_enemy = Enemy(player, 50, 18, 60)
    invincible_enemy.update_feature()
    assert not invincible_enemy.can_activated()


def test_enemy_get_feature():
    player = Player(1000, 10, 10, 0, 0)
    enemy = Enemy(player, 50, 18, 6)
    enemy.update_feature()
    assert enemy.get_feature(False) == (-96, 0, 0, 0, 0, 50, 18, 6)
    assert enemy.get_feature(True) == (-96, 0, 0, 0, 0, 50 / 12 - 4, 8, 12, -4)


def test_enemy_update_feature():
    player = Player(1000, 10, 10, 0, 0)
    enemy = Enemy(player, 50, 18, 6)
    enemy.update_feature()
    assert enemy.enemy_hp == 50
    assert enemy.enemy_atk == 18
    assert enemy.enemy_def == 6
    assert enemy.inc_hp == -96
    assert enemy.threshold_demand == 50 / 12 - 4
    assert enemy.atk_profit == 8
    assert enemy.def_profit == 12
    assert enemy.balance == -4


def test_invincible_enemy_get_feature():
    player = Player(1000, 10, 10, 0, 0)
    enemy = Enemy(player, 800, 20, 20)
    enemy.update_feature()
    assert enemy.get_feature(False) == (-1000, 0, 0, 0, 0, 800, 20, 20)
    assert enemy.get_feature(True) == (-1000, 0, 0, 0, 0, 11, 10, 0, -20)


def test_item_activate():
    player = Player(100, 5, 5, 1, 1)
    jewel = Item(player, 0, 2, 0, 0, 0)
    potion = Item(player, 50, 0, 0, 0, 0)
    key = Item(player, 0, 0, 0, 1, 0)
    door = Item(player, 0, 0, 0, 0, -1)
    jewel.activate()
    potion.activate()
    key.activate()
    door.activate()
    assert player.player_atk == 7
    assert player.player_hp == 150
    assert player.yellow_key == 2
    assert player.blue_key == 0


def test_item_can_activated():
    player = Player(100, 10, 10, 1, 0)
    yellow_door = Item(player, 0, 0, 0, -1, 0)
    assert yellow_door.can_activated()
    blue_door = Item(player, 0, 0, 0, 0, -1)
    assert not blue_door.can_activated()
    yellow_door.activate()
    assert not yellow_door.can_activated()


def test_item_get_feature():
    player = Player(100, 5, 5, 1, 1)
    jewel = Item(player, 0, 2, 0, 0, 0)
    potion = Item(player, 50, 0, 0, 0, 0)
    key = Item(player, 0, 0, 0, 1, 0)
    door = Item(player, 0, 0, 0, 0, -1)
    jewel.update_feature()
    potion.update_feature()
    key.update_feature()
    door.update_feature()
    assert jewel.get_feature(False) == (0, 2, 0, 0, 0, 0, 0, 0)
    assert potion.get_feature(False) == (50, 0, 0, 0, 0, 0, 0, 0)
    assert key.get_feature(False) == (0, 0, 0, 1, 0, 0, 0, 0)
    assert door.get_feature(False) == (0, 0, 0, 0, -1, 0, 0, 0)
    assert jewel.get_feature(True) == (0, 2, 0, 0, 0, 0, 0, 0, 0)
    assert potion.get_feature(True) == (50, 0, 0, 0, 0, 0, 0, 0, 0)
    assert key.get_feature(True) == (0, 0, 0, 1, 0, 0, 0, 0, 0)
    assert door.get_feature(True) == (0, 0, 0, 0, -1, 0, 0, 0, 0)


def test_item_update_feature():
    player = Player(100, 5, 5, 1, 1)
    jewel = Item(player, 0, 2, 0, 0, 0)
    potion = Item(player, 50, 0, 0, 0, 0)
    key = Item(player, 0, 0, 0, 1, 0)
    door = Item(player, 0, 0, 0, 0, -1)
    jewel.update_feature()
    potion.update_feature()
    key.update_feature()
    door.update_feature()
    assert jewel.inc_atk == 2
    assert potion.inc_hp == 50
    assert key.yellow_key == 1
    assert door.blue_key == -1
