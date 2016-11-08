from __future__ import division
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import operator

action_set = ['None', 'forward', 'left', 'right']

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.state = None
        self.Q_learn = {}
        self.alpha = 1 #learning rate
        self.gamma = 0.2 #future discount
        self.epsilon = 0.8 # exploration rate
        self.epsilonDecay = 0.991
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_init  = False
        self.targetReachedCount = 0
        self.iterationCount = 0
        self.rewards_history = []


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update_state(self, way_point, inputs, deadline):
        dict_key = ''

        dict_key += 'N' if way_point is None else way_point[0]
        dict_key += 'N' if inputs['light'] is None else inputs['light'][0]
        dict_key += 'N' if inputs['oncoming'] is None else inputs['oncoming'][0]
        dict_key += 'N' if inputs['right'] is None else inputs['right'][0]
        dict_key += 'N' if inputs['left'] is None else inputs['left'][0]

        self.state = dict_key

    def choose_action_greedy(self, state):
        # POLICY: EPSILON GREEDY
        eps = np.random.rand(1)
        if eps > self.epsilon: #90% of the time choose greedily
            #action = max(self.Q_learn[state],key=self.Q_learn[state].get)
            action = random.choice(
                {ac:va for ac, va in self.Q_learn[state].items()
                 if va == max (self.Q_learn[self.state].values())}.keys())
            #print (action, self.Q_learn[state])
            return ('greedy', action)
        else: #random choice #10% of the time
            return  ('random', action_set[random.randint(0,len(action_set)-1)])

    def choose_action_softmax(self, state):
        tau = 100.
        try:
            ex = np.exp(np.array([self.Q_learn[state][i] for i in action_set]) / tau)
        except KeyError:
            print 'KeyError'
            raise
        pmf = ex / np.sum(ex)
        move = action_set[np.random.choice(len(action_set), p=pmf)]
        return move

    def q_learn_stats(self):  # all state:action pairs w/ non-zero Q
        kist = [{k: v for k, v in self.Q_learn[key].iteritems()} for key in self.Q_learn.keys()]
        return {k: v for k, v in dict(zip(self.Q_learn.keys(), kist)).iteritems() if bool(v)}

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        if(t == 0):
            self.iterationCount += 1
            self.epsilon = self.epsilon * self.epsilonDecay
            print(self.epsilon)

        # TODO: Update state
        self.update_state(self.next_waypoint, inputs, deadline)
        #print self.state
        
        # TODO: Select action according to your policy
        #action = action_set[random.randint(0,len(action_set)-1)]
        if self.state not in self.Q_learn:
            self.Q_learn[self.state] = {}
            for i in action_set:
                self.Q_learn[self.state][i] = 0.
        #action = self.choose_action_softmax(self.state)
        greedy_type, action = self.choose_action_greedy(self.state)
        action_str = None if action is 'None' else action

        # Execute action and get reward
        reward = self.env.act(self, action_str)

        #store greedy actions that caused penality for analysis
        if reward < 0 and greedy_type is 'greedy':
            self.rewards_history.append([self.iterationCount, inputs, self.next_waypoint, action, reward, self.Q_learn[self.state].copy()])

        print "LearningAgent.update(): iter= {}, self.t = {}, deadline = {}, inputs = {}, waypoints ={}, action = {}, greedy_type = {}, reward = {}, q_values = {}".format(self.iterationCount, t, deadline, inputs, self.next_waypoint, action, greedy_type, reward, self.Q_learn[self.state])  # [debug]

        # TODO: Learn policy based on state, action, reward
        if self.prev_init is True:
            self.Q_learn[self.prev_state][self.prev_action] += (self.alpha * (self.prev_reward  +
                (self.gamma * self.Q_learn[self.state][max(self.Q_learn[self.state], key=self.Q_learn[self.state].get)]) - self.Q_learn[self.prev_state][self.prev_action]))

        self.prev_action = action
        self.prev_state  = self.state
        self.prev_reward  = reward
        self.prev_init  = True
        #print(self.q_learn_stats())




def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False, live_plot=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print('Negative reward states: iterationcount, input, waypoint, action, reward, q_value:')
    print(len(a.rewards_history))
    for e in a.rewards_history:
        print(e)
    print("Q_learning table:")
    print(a.q_learn_stats())
    print "Successful journeys : {}".format(a.targetReachedCount)


def run2(): #helps to find sweetspot for alpha, gammma values

    alphas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    gammas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    heatmap = []

    for i, alpha in enumerate(alphas):
        row = []
        for j, gamma in enumerate(gammas):
            e = Environment()
            a = e.create_agent(LearningAgent)
            a.alpha = alpha
            a.gamma = gamma

            e.set_primary_agent(a, enforce_deadline=True)
            sim = Simulator(e, update_delay=0.0, display=False)
            sim.run(n_trials=100)
            print "Successful journeys : {}".format(a.targetReachedCount)
            row.append(a.targetReachedCount / 100.0)
            #qstats.append(a.q_learn_stats())
        heatmap.append(row)

    print heatmap
    ax = sns.heatmap(heatmap, xticklabels=gammas, yticklabels=alphas, annot=True)
    ax.set(xlabel="gamma", ylabel="alpha")
    plt.show()

if __name__ == '__main__':
    run()
