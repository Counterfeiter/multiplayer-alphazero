import numpy as np
import sys
import copy
from scipy.signal import correlate2d
from .kingdombuildergym import ParametricCNNKingdomBuilderEnv
sys.path.append("..")
from game import Game

# Implementation for three-player Tic-Tac-Toe.
class AZKingdomBuilder(Game):

    # Returns a blank Tic-Tac-Mo board.
    # There are extra layers to represent X and O pieces, as well as a turn indicator layer.
    def get_initial_state(self, **kwargs):
        #level = 'easy'
        level = 'professional'
        train = True
        if 'train' in kwargs:
            train = kwargs['train']
        env = ParametricCNNKingdomBuilderEnv(level=level, normalize_obs=True, verbose=0, fail_actions=1, train=train)
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
            rew = -np.ones(self.get_num_players())
            score = np.array([player.score for player in s["env"].wrapped.game.players])
            #consider multiple winner
            for winner in np.argwhere(score == np.amax(score)).flatten():
                rew[winner] = 1
            return rew
        return None

    # Return 0 for X's turn or 1 for O's turn.
    def get_player(self, s):
        return int(s["env"].wrapped.game.player) - 1

    def get_current_players(self, s):
        return len(s["env"].wrapped.game.players)

    # Fixed constant for max players
    def get_num_players(self):
        return 5

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        print(s["env"].wrapped.game.board)

    def get_hash(self, s):
        return  hash(str(s["env"].wrapped.game.main_move).encode("ascii") + \
                str(s["env"].wrapped.game.old_action).encode("ascii") + \
                str(s["env"].wrapped.game.player.card).encode("ascii") + \
                str(s["env"].wrapped.game.select_coord).encode("ascii") + \
                s["obs"].tostring() + self.get_available_actions(s).tostring())
