import sys
import numpy as np
sys.path.append("..")
from mcts import MCTS
from player import Player
from mcts_ray import MCTS as MCTSRAY, Node, RootParentNode

class DeepMCTSPlayer(Player):

    def __init__(self, game, nn, simulations):
        self.game = game
        self.simulations = simulations
        self.nn = nn
        self.mcts_config = {
            "puct_coefficient": 3.0,
            "num_simulations": 20,
            "temperature": 1.5,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": True,
            "add_dirichlet_noise": False,
        }
        self.reset()

    def update_state(self, s):
        #if self.tree_node is None:
        #generate root node
        self.tree_node = Node(
            state=s,
            obs=s["obs"],
            reward=0,
            done=False,
            action=None,
            parent=RootParentNode(env=self.game, state=s),
            mcts=self.tree)

        # Think
        p, action, self.tree_node = self.tree.compute_action(self.tree_node)
        available = self.game.get_available_actions(s)
        template = np.zeros_like(available)
        template[action] = 1
        s_prime = self.game.take_action(s, template)
        return s_prime

    def reset(self):
        self.tree_node = None
        self.tree = MCTSRAY(self.nn, self.mcts_config)
