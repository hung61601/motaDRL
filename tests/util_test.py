from env.util import condition_clamp


def test_condition_clamp():
    assert condition_clamp(40, 50, 30) == 40
    assert condition_clamp(70, 50, 30) == 50
    assert condition_clamp(10, 50, 30) == 30
