import numpy as np
import sys
import copy
from scipy.signal import correlate2d
from .kingdombuildergym import ParametricCNNKingdomBuilderEnv
sys.path.append("..")
from game import Game

scalernorm = lambda xarr : [((x - min(xarr)) / (max(xarr) - min(xarr)) * (1. - (-1.)) + (-1.)) for x in xarr]

# Implementation for three-player Tic-Tac-Toe.
class AZKingdomBuilder(Game):

    # Returns a blank Tic-Tac-Mo board.
    # There are extra layers to represent X and O pieces, as well as a turn indicator layer.
    def get_initial_state(self, **kwargs):
        #level = 'easy'
        #level = 'intermediate'
        level = 'professional'
        #level = 'professional2player'
        train = True
        if 'train' in kwargs:
            train = kwargs['train']
        print("Game is in " + ("train state" if train else "evaluation state"))
        env = ParametricCNNKingdomBuilderEnv(level=level, normalize_obs=True, verbose=0, fail_actions=1, train=train)
        if 'loadgamefile' in kwargs:
            obs = env.reset(kwargs['loadgamefile'])
        else:
            obs = env.reset()
        return {"env":env, "obs":obs["obs"]}

    # Returns a 3x5 boolean ndarray indicating open squares. 
    def get_available_actions(self, s):
        return s["env"].action_mask
        
    # Place an X, O, or Y in state.
    def take_action(self, s, a):
        s_copy = copy.deepcopy(s)
        action = np.argmax(a)
        obs, _, _, _ = s_copy["env"].step(action)
        s_copy["obs"] = obs["obs"]
        return s_copy

   # Check all possible 3-in-a-rows for a win.
    def check_game_over(self, s):
        if s["env"].done:
            return True
        return False
    
    # if you have a game where the winner is unknown until the game is done
    # return None until game is done
    def get_scores(self, s):

        # if s["env"].done:
        score = s["env"].game.rules.score(s["env"].game.players)
        #     print("Game ends with score:" , score)
        # else:
        #     return None
        rew = -np.ones(self.get_num_players())
        #consider multiple winner
        #for winner in np.argwhere(score == np.amax(score)).flatten():
        #    rew[winner] = 1

        #print("Games ends: ", rew)
        if min(score) == max(score):
            #all player playing this match wins because all have equal score or zero score at start
            return np.append(np.ones(len(score)), -np.ones(self.get_num_players() - len(score)))

        for i, score in enumerate(scalernorm(score)):
            rew[i] = score
        return rew

    # Return 0 for X's turn or 1 for O's turn.
    def get_player(self, s):
        return int(s["env"].game.player) - 1

    def get_current_players(self, s):
        return len(s["env"].game.players)

    # Fixed constant for max players
    def get_num_players(self):
        return 5

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        s["env"].render()

    def get_hash(self, s):
        return  hash(str(s["env"].game.main_move).encode("ascii") + \
                str(s["env"].game.old_action).encode("ascii") + \
                str(s["env"].game.player.card).encode("ascii") + \
                str(s["env"].game.select_coord).encode("ascii") + \
                str(s["env"].game.townstoplay).encode("ascii") + \
                s["obs"].tostring() + self.get_available_actions(s).tostring())
