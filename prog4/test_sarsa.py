import numpy as np
import gym
from sarsa import SarsaLambda, StateActionFeatureVectorWithTile

def test_sarsa_lamda():
    env = gym.make("MountainCar-v0")
    gamma = 1.

    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=10,
        tile_width=np.array([.45,.035])
    )

    w = SarsaLambda(env, gamma, 0.8, 0.01, X, 2000)

    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(render=False):
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s,done)
            s,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G

    Gs = [_eval() for _ in  range(100)]
    _eval(True)

    assert np.max(Gs) >= -110.0, 'fail to solve mountaincar'

if __name__ == "__main__":
    test_sarsa_lamda()
