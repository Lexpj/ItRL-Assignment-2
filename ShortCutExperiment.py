# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from ShortCutAgents import *
from ShortCutEnvironment import *
import sys
import matplotlib.pyplot as plt

def run_repetitions():
    n_episodes = 10000
    n_rep = 100
    res = np.empty((n_rep, n_episodes))
    
    for rep in range(n_rep):
        env = ShortcutEnvironment()
        agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=0.1)
        
        for episode in range(n_episodes):
            sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
            sys.stdout.flush()

            state = env.state()
            while not env.done():
                action = agent.select_action(state)
                reward = env.step(action)
                res[rep][episode] += reward
                stateprime = env.state()
                agent.update(state,action,reward,stateprime)
                state = stateprime
            env.reset()
                
    
    mean_res = res.mean(axis = 0)
    print(res,mean_res)
    plt.plot(mean_res)
    plt.show()


run_repetitions()