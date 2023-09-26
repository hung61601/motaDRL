import time


def test():
    run_time = time.time()
    for _ in range(100000):
        a = [i for i in range(100)]
        a.pop(0)
    t = time.time() - run_time
    print('\ntime: {0:.5f} s'.format(t))


def test02():
    candidate_events = {i for i in range(5000)}
    candidate_list = [i for i in range(5000)]
    run_time = time.time()
    for _ in range(10000):
        candidate_events.add(30)
        # list(candidate_events)
        # list(candidate_events)
        # candidate_events.remove(30)
        # candidate_list.append(30)
        # candidate_list.remove(30)
    t = time.time() - run_time
    print(t)


def test03():
    import collections
    node_links = collections.OrderedDict()
    node_links[50] = 30
    print(node_links.pop(50))


def test04():
    import numpy as np
    masked = np.array([True, False, True, False])
    weights = ~masked
    normalized = weights.ravel() / float(weights.sum())
    print(weights)
    print(np.random.choice(np.arange(masked.size), 1, p=normalized))



def test_random():
    from env.mota_env import Mota
    from env.mota_builder import MotaBuilder
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
    num_epochs = 10000
    total_score = 0
    import copy
    import time
    import numpy as np
    start_time = time.time()
    for i in range(num_epochs):
        done = False
        _, info = env.reset(*copy.deepcopy(mota_data))
        while not done:
            masked = np.array(info['mask'])
            weights = ~masked
            normalized = weights.ravel() / float(weights.sum())
            action = np.random.choice(np.arange(masked.size), 1, p=normalized)
            _, reward, done, _, info = env.step(int(action))
        total_score += env.player.player_hp
    print('avg. score:', total_score / num_epochs)
    print('run time:', time.time() - start_time)
