"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
import collections
import math

import numpy as np


class Node:
    def __init__(self, action, obs, done, reward, state, mcts, parent=None):
        self.env = parent.env
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.valid_actions = self.env.get_available_actions(state)
        self.current_player = self.env.get_player(state)
        self.action_space_size = self.valid_actions.shape[0]
        self.child_total_value = np.zeros(
            [self.action_space_size, self.env.get_num_players()], dtype=np.float32)  # Q
        self.child_priors = np.zeros(
            [self.action_space_size, self.env.get_num_players()], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32)  # N


        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        if type(self.parent.child_total_value) == collections.defaultdict:
            return self.parent.child_total_value[self.action]
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        if type(self.parent.child_total_value) == collections.defaultdict:
            self.parent.child_total_value[self.action] = value
        else:
            self.parent.child_total_value[self.action] = value

    def child_Q(self):
        # TODO (weak todo) add "softmax" version of the Q-value
        return self.child_total_value[:,self.parent.current_player] / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * self.child_priors / (1 + self.child_number_visits)

    def best_action(self):
        """
        :return: action
        """
        u = self.child_U()
        q = self.child_Q()
        child_score = q + self.mcts.c_puct * u
        masked_child_score = child_score
        masked_child_score[~self.valid_actions] = -np.inf
        return np.argmax(masked_child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            template = np.zeros(self.action_space_size) # Submit action to get s'
            template[action] = 1.0
            next_state = self.env.take_action(self.state, template)
            
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=self.env.get_scores(next_state),
                done=self.env.check_game_over(next_state),
                obs=next_state["obs"],
                mcts=self.mcts)
            
            #store reference to the node
            self.mcts.lockup_table = { self.env.get_hash(next_state) : self.children[action] }
        return self.children[action]

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value
            current = current.parent


class RootParentNode:
    def __init__(self, env, state):
        self.parent = None
        #state = env.get_initial_state()
        #action_space_size = env.get_available_actions(state).shape[0]
        self.current_player = env.get_player(state)
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.env = env


class MCTS:
    def __init__(self, model, mcts_param):
        self.model = model
        self.temperature = mcts_param["temperature"]
        self.dir_epsilon = mcts_param["dirichlet_epsilon"]
        self.dir_noise = mcts_param["dirichlet_noise"]
        self.num_sims = mcts_param["num_simulations"]
        self.exploit = mcts_param["argmax_tree_policy"]
        self.add_dirichlet_noise = mcts_param["add_dirichlet_noise"]
        self.c_puct = mcts_param["puct_coefficient"]

        self.lockup_table = {}

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.predict_ray(leaf.obs, leaf.valid_actions)
                if self.add_dirichlet_noise:
                    child_priors = (1 - self.dir_epsilon) * child_priors
                    noise = np.random.dirichlet([self.dir_noise] * child_priors.size)
                    child_priors += self.dir_epsilon * noise

                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)
        
        tree_policy = np.power(node.child_number_visits, self.temperature) / node.number_visits
        #print(tree_policy)
        max_policy = np.max(tree_policy)
        if max_policy == 0.0:
            return None, None, None
        tree_policy = tree_policy / np.max(tree_policy)  # to avoid overflows when computing softmax
        #print(tree_policy)
        #tree_policy = np.power(tree_policy, self.temperature)
        tree_policy = tree_policy / np.sum(tree_policy)
        #print(tree_policy)
        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            action = np.argmax(tree_policy)

            if node.valid_actions[action] == False:
                print("This action is not avalibale (rand in game flow) -> start search again")
                return self.compute_action(node)
        else:
            # otherwise sample an action according to tree policy probabilities
            #bestAs = np.array(np.argwhere(tree_policy > 0.0)).flatten()
            #action = np.random.choice(bestAs)
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]
