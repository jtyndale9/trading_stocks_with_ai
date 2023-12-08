""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  

Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students	 		 		   		 		  
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a 
GT honor code violation.

-----do not edit anything above this line---

Student Name: Joshua Tyndale
GT User ID: jtyndale3
GT ID: 903767547
"""

import random as rand
import numpy as np

class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha # learning rate: how much dop you trust new info? Between 0 and 1
        self.gamma = gamma # what is the value of future rewards?
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q = np.zeros((num_states, num_actions))

        if dyna != 0:
            self.T = np.zeros((num_states, num_actions, num_states))
            self.Tc = [[[0.00001 for k in range(num_states)] for j in range(num_actions)] for i in range(num_states)]
            self.R = np.zeros((num_states, num_actions))

    def author(self):
        return 'jtyndale3'

    def querysetstate(self, s):
        self.s = s
        action = rand.randint(0, self.num_actions - 1)
        self.a = action
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def get_rand_float(self):
        return rand.uniform(0, 1)

    def update_t_r(self, last_state, last_action, reward):
        self.Tc[last_state][last_action][self.s] += 1
        self.R[last_state][last_action] = (1 - self.alpha) * self.R[last_state][last_action] + self.alpha * reward
        self.T[last_state][last_action][self.s] = self.Tc[last_state][last_action][self.s] / sum(self.Tc[last_state][last_action])


    def hallucinate(self):

        # random s
        random_state = rand.randint(0, self.num_states - 1)
        # random a
        random_action = rand.randint(0, self.num_actions - 1)

        next_state = np.argmax(self.T[random_state][random_action])
        reward = self.R[random_state][random_action]
        if (self.T[random_state][random_action][next_state] == 0):
            return

        #  find best action to take, given the current state:
        next_move = np.argmax(self.Q[random_state])
        self.Q[random_state][random_action] = (1 - self.alpha) * (self.Q[random_state][random_action]) + self.alpha * (
            reward + self.gamma * self.Q[next_state][next_move])

    def query(self, s_prime, r):

        last_state = self.s
        last_action = self.a
        current_state = s_prime
        reward = r

        # choose a random action with probability rar, call random number generator between 1 and 0, if num is less than rar, do an action
        if self.get_rand_float() < self.rar: # do a random action if we hit it
            next_move = rand.randint(0, self.num_actions - 1)

        else:
            # find best action to take, given the current state:
            # ie: find the max value of the Q table at index 'state'
            next_move = np.argmax(self.Q[current_state])

        # use the new information s_prime and r to update the Q table
        self.Q[last_state][last_action] = (1 - self.alpha) * (self.Q[last_state][last_action]) + self.alpha * (
                    reward + self.gamma * self.Q[current_state][next_move])

        # Update current values:
        self.a = next_move
        self.s = current_state

        # Update rar according to the decay rate radr at each step
        self.rar = self.rar * self.radr

        if self.verbose:
            print(f"s = {s_prime}, a = {next_move}, r={r}")

        if self.dyna != 0:
            self.update_t_r(last_state, last_action, reward)
            for i in range(self.dyna):
                self.hallucinate()

        # query() should return an integer, which is the next action to take
        return next_move

    def just_query(self, state):
        #print(state)
        return np.argmax(self.Q[state])

if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")

