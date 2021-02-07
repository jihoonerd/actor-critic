import os
import pathlib

import gym
import imageio
import numpy as np
import pyvirtualdisplay
import matplotlib.pyplot as plt

from ac.agent import Agent

if __name__ == '__main__':
    display = pyvirtualdisplay.Display(visible=False)
    display.start()
    pathlib.Path('assets').mkdir(parents=True, exist_ok=True)

    env = gym.make('CartPole-v1')
    agent = Agent(env=env)
    num_episodes = 2000

    best_score = env.reward_range[0]
    score_history = []

    for i in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        frame_set = []

        while not done:
            action = agent.choose_action(state)
            action = 1 if action.numpy() > 0 else 0  # Ensure action to be either 0 or 1
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.learn(state, action, reward, next_state, done)

            if i % 100 == 0:
                frame_set.append(env.render(mode='rgb_array'))

            state = next_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if i % 100 == 0:
            imageio.mimsave(os.path.join(
                'assets', f'eps-{i}.gif'), frame_set, fps=30)
            plt.figure()
            plt.plot(score_history, label='score')
            plt.legend(loc='upper left')
            plt.xlabel('epsiode')
            plt.ylabel('score')
            plt.grid()
            plt.savefig(os.path.join('assets', 'score_fig.png'))

        
        print(f'Episode: {i} / Score: {score} / 100 Episode AVG: {avg_score}')
