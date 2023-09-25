import time


def test_clamp():
    def max_min_clamp(largest, n, smallest):
        return max(min(largest, n), smallest)

    def sorted_clamp(largest, n, smallest):
        return sorted([largest, n, smallest])[1]

    def condition_clamp(largest, n, smallest):
        return smallest if smallest > n else n if largest > n else largest

    print()
    run_time = time.time()
    for _ in range(100000):
        max_min_clamp(50, 40, 30)
        max_min_clamp(50, 70, 30)
        max_min_clamp(50, 10, 30)
    t = time.time() - run_time
    print('  max_min_clamp: {0:.5f} s'.format(t))
    run_time = time.time()
    for _ in range(100000):
        sorted_clamp(50, 40, 30)
        sorted_clamp(50, 70, 30)
        sorted_clamp(50, 10, 30)
    t = time.time() - run_time
    print('   sorted_clamp: {0:.5f} s'.format(t))
    run_time = time.time()
    for _ in range(100000):
        condition_clamp(50, 40, 30)
        condition_clamp(50, 70, 30)
        condition_clamp(50, 10, 30)
    t = time.time() - run_time
    print('condition_clamp: {0:.5f} s'.format(t))
