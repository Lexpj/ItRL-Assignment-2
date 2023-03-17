# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
from ShortCutAgents import *
from ShortCutEnvironment import *
import sys
import matplotlib.pyplot as plt


def render_policy(env, agent):
    print()
    for r in range(env.r):
        for c in range(env.c):
            if env.s[r][c] == "X":
                a = np.argmax(agent.Q[r*env.c + c])
                if a == 0:
                    print("^",end="")
                elif a==1:
                    print("v",end="")
                elif a==2:
                    print("<",end="")
                elif a==3:
                    print(">",end="")
            else:
                print(env.s[r][c],end="")
        print()
    print()

def plot(res):
    mean_res = res.mean(axis = 0)
    print(res,mean_res)
    plt.plot(mean_res)
    plt.show()

def run_repetitions(
        agent_type,
        windy = False, 
        n_episodes=10000,
        n_rep = 100,
        alpha = 0.1,
        gamma = 1.0,
        epsilon = 0.1):
    
    res = np.empty((n_rep, n_episodes))
    env = WindyShortcutEnvironment() if windy else ShortcutEnvironment()

    if agent_type == "qlearning":

        for rep in range(n_rep):
            agent = QLearningAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon)
            for episode in range(n_episodes):
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.state()
                # Loop for each step of episode, until S is terminal
                while not env.done():
                    # Choose A from S using egreedy derived from Q 
                    action = agent.select_action(state)
                    # Take action A, observe R, S'
                    reward = env.step(action)
                    res[rep][episode] += reward
                    stateprime = env.state()
                    # Q(S,A) = Q(S,A) + alpha(R + gamma*max_a(Q(S',A)) - Q(S,A))
                    agent.update(state,action,reward,stateprime,alpha=alpha,gamma=gamma)
                    # S = S'
                    state = stateprime
                # Reset environment
                env.reset()  
        

    elif agent_type == "sarsa":
        
        for rep in range(n_rep):
            agent = SARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon)
            for episode in range(n_episodes):
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.state()
                # Choose A from S using egreedy derived from Q 
                action = agent.select_action(state)
                # Loop for each step of episode, until S is terminal
                while not env.done():
                    # Take action A, observe R, S'
                    reward = env.step(action)
                    res[rep][episode] += reward
                    stateprime = env.state()
                    # Choose A' from S' using egreedy derived from Q
                    actionprime = agent.select_action(stateprime)
                    # Q(S,A) = Q(S,A) + alpha(R + gamma * Q(S',A') - Q(S,A))
                    agent.update(state,action,reward,stateprime,actionprime,alpha=alpha,gamma=gamma)
                    # S = S'
                    state = stateprime
                    # A = A'
                    action = actionprime
                # Reset environment
                env.reset()      
    
    
    elif agent_type == "expectedsarsa":

        for rep in range(n_rep):
            agent = ExpectedSARSAAgent(n_actions=env.action_size(), n_states=env.state_size(), epsilon=epsilon)
            for episode in range(n_episodes):
                sys.stdout.write("\r [" + "="*int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20) + "."*(20-(int(((episode+1+(rep*n_episodes))/(n_episodes*n_rep)) * 20))) +f"] {episode+1+(rep*n_episodes)}/{(n_episodes*n_rep)}")
                sys.stdout.flush()

                # Initialize S
                state = env.state()
                # Loop for each step of episode, until S is terminal
                while not env.done():
                    # Choose A from S using egreedy derived from Q 
                    action = agent.select_action(state)
                    # Take action A, observe R, S'
                    reward = env.step(action)
                    res[rep][episode] += reward
                    stateprime = env.state()
                    # Q(S,A) = Q(S,A) + alpha(R + gamma*sum_{A'}(p(A'|S')*Q(S',A')) - Q(S,A))
                    agent.update(state,action,reward,stateprime,alpha=alpha,gamma=gamma)
                    # S = S'
                    state = stateprime
                # Reset environment
                env.reset()        
           
    return res


def main():

    pass
