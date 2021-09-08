import numpy as np
import matplotlib.pyplot as plt
import random
from gridworld import * # Get this from GitHub!

def print_qtable(q_table, casenumber):
    f = open("q_table-" + str(casenumber) + ".txt", "w")
    for i in range(q_table.shape[0]):
        for j in range(q_table.shape[1]):
            f.write("(")
            f.write(str(i+1))
            f.write("," + str(j+1))
            f.write(")\n")
            f.write("   up:" + str(q_table[i,j,0]) + "\n")
            f.write("   left:" + str(q_table[i,j,1]) + "\n")
            f.write("   down:" + str(q_table[i,j,2]) + "\n")
            f.write("   right:" + str(q_table[i,j,3]) + "\n")
            f.write("-----------------------------------------------------\n")
            
def q_learning(num_episodes, lr, gamma, eps, initialQ, casenumber):
    np.random.seed(1234)
    env = gridworld()
    env.show_grid()

    if initialQ == 0:
        q_table = np.zeros((env.dim[0], env.dim[1], len(env.action_space)))
    else:
        q_table = np.ones((env.dim[0], env.dim[1], len(env.action_space)))*20

    ep_rewards = []
    n_steps = []

    for ep in range(num_episodes):
        s_0 = env.reset()
        done = False
        rewards = 0
        while done == False:
            # Take random action with epsilon probability
            if np.random.rand() < eps:
                action = np.random.choice(env.action_space)
            else:
                # Take greedy action
                action = np.argmax(q_table[s_0[0], s_0[1]])
            if action == 0:
                if random.random() >= 0.6:
                    action = 2 
            s_1, reward, done = env.step(action)

            # Update the Q-table
            q_table[s_0[0], s_0[1], action] += lr*(reward + \
                                                   gamma*np.max(q_table[s_1[0], s_1[1]]) \
                                                   - q_table[s_0[0], s_0[1], action])
            s_0 = s_1.copy()
            rewards += reward
            if casenumber == 3:
                if env.n == 100:
                    done = True
            if done:
                ep_rewards.append(rewards)
                

    # Calculate rolling average
    mean_rewards = [np.mean(ep_rewards[n-10:n]) if n > 10 else np.mean(ep_rewards[:n]) 
                   for n in range(1, len(ep_rewards))]
    print_qtable(q_table,casenumber)
    # Plot results
    plt.figure(figsize=(12,8))
    plt.plot(ep_rewards)
    plt.plot(mean_rewards)
    plt.title('Gridworld Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('reward-episod-' + str(casenumber))

    env.plot_policy(q_table, casenumber=casenumber)
states = [[1000,0.1,0.5,0.2,0,1],
        [1000,0.1,0.5,0.2,20,2],
        [1000,0.1,0.5,0,0,3],
        [1000,0.1,0.1,0.2,0,4],
        [1000,0.9,0.5,0.2,0,5]]
n = 1
for state in states: 
    print("state {}".format(n))
    q_learning(state[0], state[1], state[2], state[3], state[4], state[5])
    n += 1