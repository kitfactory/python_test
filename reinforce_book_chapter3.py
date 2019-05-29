#### 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import gym


ENV = 'CartPole-v0'
NUM_DIZITIZED = 6



frames = []
env = gym.make(ENV)
observation = env.reset()


# linspaceで等差数列、その前後ろを削った配列を返却する

def bins(clip_min,clip_max,num):
    ls = np.linspace(clip_min,clip_max, num+1)
    print(ls)
    return ls[1:-1]


def dizitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    dizitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIZITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIZITIZED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIZITIZED))
    ]
    return sum([x*(NUM_DIZITIZED**i) for i, x in enumerate(dizitized)])

fig = plt.figure()

for step in range(0,200):
    img = env.render(mode='rgb_array')
    #    print(img.__class__)
    frames.append(img)
    action = np.random.choice(2)
    observation, reward, done, info = env.step(action)


def show_animation_image(frames):
    ims = []
    for img in frames:
        im = plt.imshow(img,animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
                                repeat_delay=100)
    plt.show()

show_animation_image(frames)
