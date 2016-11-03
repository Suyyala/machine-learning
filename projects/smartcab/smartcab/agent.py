import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

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
        self.q_value = {}
        self.alpha = 1 #learning rate
        self.gamma = 0. #future discount
        self.epsilon = 0.1 # exploration rate
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_init  = False


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
            action = max(self.Q_learn[state], key=self.Q_learn[state].get)
            return action
        else: #random choice #10% of the time
            return action_set[np.random.choice(len(action_set))]

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
        kist = [{k: v for k, v in self.Q_learn[key].iteritems() if v != 0} for key in self.Q_learn.keys()]
        return {k: v for k, v in dict(zip(self.Q_learn.keys(), kist)).iteritems() if bool(v)}

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

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
        action = self.choose_action_greedy(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action=None if action is 'None' else action)


        # TODO: Learn policy based on state, action, reward
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        if self.prev_init is True:
            self.Q_learn[self.prev_state][self.prev_action] += (self.alpha * (self.prev_reward +
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
    sim = Simulator(e, update_delay=0.5, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
