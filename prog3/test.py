import numpy as np
import gym
from algo import semi_gradient_n_step_td
from policy import Policy
from tc import ValueFunctionWithTile
from nn import ValueFunctionWithNN

testing_states = np.array([[-.5, 0], [-0.2694817,  0.014904 ], [-1.2,  0. ], [-0.51103601,  0.06101282], [ 0.48690072,  0.04923175]])
correct_values = np.array([-108, -76.5, -38.5, -18.2, -2])

def test_tile():
    env = gym.make("MountainCar-v0")
    gamma = 1.

    policy = Policy()
    V = ValueFunctionWithTile(
        env.observation_space.low,
        env.observation_space.high,
        num_tilings=10,
        tile_width=np.array([.45,.035]))

    semi_gradient_n_step_td(env,1.,policy,10,0.01,V,1000)

    Vs = [V(s) for s in testing_states]
    print(Vs)
    assert np.allclose(Vs,correct_values,1e-2,3), f'{correct_values} != {Vs}, but it might due to stochasticity'

def test_nn():
    env = gym.make("MountainCar-v0")
    gamma = 1.

    policy = Policy()
    V = ValueFunctionWithNN(env.observation_space.shape[0])

    semi_gradient_n_step_td(env,1.,policy,10,0.001,V,1000)

    Vs = [V(s) for s in testing_states]
    print(Vs)
    assert np.allclose(Vs,correct_values,0.20,5.), f'{correct_values} != {Vs}, but it might due to stochasticity'

if __name__ == "__main__":
    test_tile()
    test_nn()
