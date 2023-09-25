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