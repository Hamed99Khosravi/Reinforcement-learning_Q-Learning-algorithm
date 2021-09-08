import matplotlib.pyplot as plt
import numpy as np
import random

class gridworld:
    
    def __init__(self):
        self.dim = [4, 5]

        self.pos_goal = [0, 0]
        self.reward_goal = 10
        
        self.pos_trap = [1, 2]
        self.reward_trap = -10
        
        self.repair = [0, 1]
        
        self.start = self.random_start()
        
        self.s = self.start[:]
        
        self.walls = [[0,1],[1,1],[2,1],[0,2]]
        self.reward_wall = -1
        self.reward_out = -1
        self.complete = False
        self.harmness = 0
        
        self.n = 0
        self.action_space = [0, 1, 2, 3]
        self.action_dict = {'Up': 0,
                           'Left': 1,
                           'Down': 2,
                           'Right': 3}
        self.action_prob = [0.25, 0.25, 0.25, 0.25]

    def random_start(self):
        x = random.randint(0,self.dim[0]-1)
        y = random.randint(0,self.dim[1]-1)
        while [x,y] in [self.pos_goal,self.pos_trap, self.repair]:
            x = random.randint(0,self.dim[0]-1)
            y = random.randint(0,self.dim[1]-1)
        return [x,y]

    def show_grid(self):
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.pos_goal[0] and j == self.pos_goal[1]:
                    row.append("|\033[0;37;42m G \033[0;37;48m")
                elif i == self.pos_trap[0] and j == self.pos_trap[1]:
                    row.append("|\033[0;33;46m H \033[0;37;48m")
                elif i == self.repair[0] and j == self.repair[1]:
                    row.append("\033[1;31;40m|\033[0;37;40m R ")
                elif i == self.start[0] and j == self.start[1]:
                    if (i == 1 and j == 1) or (i == 2 and j == 1) or (i == 0 and j == 2):
                        row.append("\033[1;31;40m|\033[1;35;40m S \033[0;37;40m")
                    else:
                        row.append("|\033[1;35;40m S \033[0;37;40m")
                elif (i == 1 and j == 1) or (i == 2 and j == 1) or (i == 0 and j == 2):
                    row.append("\033[1;31;40m|   \033[0;37;40m")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
        
    def show_state(self):
        for i in range(self.dim[0]):
            print("-" * (self.dim[1] * 5 + 1))
            row = []
            for j in range(self.dim[1]):
                if i == self.s[0] and j == self.s[1]:
                    if (i == 0 and j == 1) or (i == 1 and j == 1) or (i == 2 and j == 1) or (i == 0 and j == 2):
                        row.append("\033[1;31;40m|\033[0;37;40mX  ")
                    else:
                        row.append("|X  ")
                elif i == self.pos_goal[0] and j == self.pos_goal[1]:
                    row.append("|\033[0;37;42m G \033[0;37;48m")
                elif i == self.pos_trap[0] and j == self.pos_trap[1]:
                    row.append("|\033[0;33;46m H \033[0;37;48m")
                elif (i == 0 and j == 1) or (i == 1 and j == 1) or (i == 2 and j == 1) or (i == 0 and j == 2):
                    row.append("\033[1;31;40m|   \033[0;37;40m")
                else:
                    row.append("|   ")
            row.append("|  ")
            print(' '.join(row))
        print("-" * (self.dim[1] * 5 + 1))
        
    # Give the agent an action
    def step(self, a):
        wall_reward, wall_check = self.checkWall(self.s,a)
        out_reward, out_check = self.checkOut(self.s,a)
        if a not in self.action_space:
            return "Error: Invalid action submission"
        # Check for special terminal states
        if self.s == self.pos_goal:
            self.complete = True
            reward = self.reward_goal
        elif self.s == self.pos_trap:
            self.complete = True
            self.harmness += 1
            if self.harmness == 2:
                reward = self.reward_trap
            else:
                reward = 0
        elif wall_check:
            reward = wall_reward
        elif out_check:
            reward = out_reward
        else:
            if self.s == self.repair:
                self.harmness = 0
            # Move up
            if a == 0 and self.s[0] > 0:    
                self.s[0] -= 1
            # Move left
            elif a == 1 and self.s[1] > 0:
                self.s[1] -= 1
            # Move down
            elif a == 2 and self.s[0] < self.dim[0] - 1:
                self.s[0] += 1
            # Move right
            elif a == 3 and self.s[1] < self.dim[1] - 1:
                self.s[1] += 1
            reward = 0
        self.n += 1
        return self.s, reward, self.complete

    def checkWall(self,s,a):
        reward = 0
        check = False
        if s == self.walls[0]:
            if a == 1 or a == 3:
                reward = self.reward_wall
                check = True
        elif s == self.walls[1]:
            if a == 1:
                reward = self.reward_wall
                check = True
        elif s == self.walls[2]:
            if a == 1:
                reward = self.reward_wall
                check = True
        elif s == self.subtractState(self.walls[1],[0,1]):
            if a == 3:
                reward = self.reward_wall
                check = True
        elif s == self.subtractState(self.walls[2],[0,1]):
            if a == 3:
                reward = self.reward_wall
                check = True
        elif s == self.walls[3]:
            if a == 1:
                reward = self.reward_wall
                check = True
        return reward, check
    
    def checkOut(self,s,a):
        reward = 0
        check = False 
        if s[0] == 0:
            if a == 0:
                reward = self.reward_out
                check = True
        elif s[0] == self.dim[0] - 1:
            if a == 2:
                reward = self.reward_out
                check = True
        elif s[1] == 0:
            if a == 1:
                reward = self.reward_out
                check = True
        elif s[1] == self.dim[1] - 1:
            if a == 3:
                reward = self.reward_out
                check = True
        return reward,check
    
    def reset(self):
        self.s = self.start[:]
        self.complete = False
        self.n = 0
        return self.s

    def subtractState(self,s1,s2):
        return [x1 - x2 for (x1, x2) in zip(s1, s2)]

    # Plots policy from q-table
    def plot_policy(self, q_table, figsize=(12,8), title='Learned Policy', casenumber=0):
        x = np.linspace(0, self.dim[1] - 1, self.dim[1]) + 0.5
        y = np.linspace(self.dim[0] - 1, 0, self.dim[0]) + 0.5
        X, Y = np.meshgrid(x, y)
        zeros = np.zeros(self.dim)

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        # Get max values
        q_max = q_table.max(axis=2)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                q_star = np.zeros(self.dim)
                q_max_s = q_max[i, j]
                max_vals = np.where(q_max_s==q_table[i,j])[0]
                for action in max_vals:
                    q_star[i,j] = 0.4
                    # Plot results
                    if action == 0:
                        # Move up
                        plt.quiver(X, Y, zeros, q_star, scale=1, units='xy')
                    elif action == 1:
                        # Move left
                        plt.quiver(X, Y, -q_star, zeros, scale=1, units='xy')
                    elif action == 2:
                        # Move down
                        plt.quiver(X, Y, zeros, -q_star, scale=1, units='xy')
                    elif action == 3:
                        # Move right
                        plt.quiver(X, Y, q_star, zeros, scale=1, units='xy')
                        
        plt.xlim([0, self.dim[1]])
        plt.ylim([0, self.dim[0]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(title)
        plt.grid()
        plt.savefig('policy-'+str(casenumber))


