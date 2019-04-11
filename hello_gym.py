import gym

env=gym.make("CartPole-v0")

for episode in range(20):
    env.reset()
    for t in range(100):
        action=env.action_space.sample()
        env.step(action)
        env.render()
        