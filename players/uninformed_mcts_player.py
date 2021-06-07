import sys
import numpy as np
sys.path.append("..")
from mcts import MCTS
from player import Player
from models.dumbnet import DumbNet
from mcts_ray import MCTS as MCTSRAY, Node, RootParentNode
from neural_network import NeuralNetwork

class UninformedMCTSPlayer(Player):

    def __init__(self, game, simulations):
        self.game = game
        self.simulations = simulations
        #self.tree = MCTS(game, NeuralNetwork(game, DumbNet), add_noise=False)
        self.mcts_config = {
            "puct_coefficient": 5.0,
            "num_simulations": simulations,
            "temperature": 1.5,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": True,
            "add_dirichlet_noise": False,
        }
        self.reset()



    def update_state(self, s):

        # maybe the game state is still in 
        if self.tree is not None and self.game.get_hash(s) in self.tree.lockup_table:
            self.tree_node = self.tree.lockup_table[self.game.get_hash(s)]
            print("#### Happy while reloading a already searched state")
        else:
            #clear memory restart with mcts search
            self.tree = MCTSRAY(NeuralNetwork(self.game, DumbNet), self.mcts_config, record_score=True, use_current_score = True)
            self.tree_node = Node(
                state=s,
                reward=0,
                done=False,
                action=None,
                parent=RootParentNode(env=self.game, state=s),
                mcts=self.tree)

        # Think
        p, action, next_nodes = self.tree.compute_action(self.tree_node)

        i = 0
        for key, nodes in self.tree_node.children.items():
            for value in nodes:
                i += 1
                print("Visit child action {}! visits: {} value: {} score: {}".format(key, value.number_visits, value.total_value / value.number_visits, value.total_reward / value.number_visits))

        available = self.game.get_available_actions(s)
        template = np.zeros_like(available)
        template[action] = 1
        s_prime = self.game.take_action(s, template)

        for node in next_nodes:
            if s_prime["env"].game.gamestate_to_dict() == node.state["env"].game.gamestate_to_dict():
                self.tree_node = node
                break

        return s_prime

    def reset(self):
        self.tree = None
