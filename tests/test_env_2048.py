import gym
import site
import numpy as np
site.addsitedir('/home/patrick/projects/IA/my-2048/src')
import gy2048
self = gym.make('2048-v0')
board = self.board


def test_reset():
    env = gym.make('2048-v0')
    env.reset()
    zero_locs = np.argwhere(env.board == 0)
    assert len(zero_locs) == 14


def test_done():
    env = gym.make('2048-v0')
    env.reset()
    assert not env.is_done()
    board = np.array(list(range(2, 18))).reshape(4, 4)
    env.board = board
    assert env.is_done()


#================================
if __name__ == '__main__':
    test_reset()
    test_done()
