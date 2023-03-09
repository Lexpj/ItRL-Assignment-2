# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from ShortCutAgents import *
from ShortCutEnvironment import *
import sys

def run_repetitions():
    n_episodes = 10000
    alpha = 0.1
    gamma = 0.1

    env = ShortcutEnvironment()


    agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1)
    
    for episode in range(n_episodes):
        sys.stdout.write("\r [" + "="*int(((episode+1)/n_episodes) * 20) + "."*(20-(int(((episode+1)/n_episodes) * 20))) +f"] {episode+1}/{n_episodes}")
        sys.stdout.flush()
        state = env.state()
        while not env.done():
            action = agent.select_action(state)
            reward = env.step(action)
            agent.update(state,action,reward)
            state = env.state()
        
run_repetitions()
        
