import math
import numpy as np
import copy
import sys

# Concerns: Add epsilon amount to UCB evaluation to ensure probability is considered
# Caveat: Q in heuristic might obviate this.
# Concerns: No Dir noise being added. If it is added, tests would break.
# Caveat: Make Dir a switch, write tests that use Dir with fixed seed.

MCTS_STOAGE_INDEX_BESTACTION = 0
MCTS_STOAGE_INDEX_N = 1
MCTS_STOAGE_INDEX_Q = 2
MCTS_STOAGE_INDEX_P = 3

# An efficient, vectorized Monte Carlo tree search implementation.
# Uses no loops, done completely with numpy.
class MCTS():

    def __init__(self, game, nn, add_noise = False):
        self.game = game
        self.nn = nn
        self.tree = {}
        self.add_noise = False

    # Produces a hash-friendly representation of an ndarray.
    # This is used to index nodes in the accumulated Monte Carlo tree.
    # forward to the game, because of possbile hidden game states a hash
    # over the oberservations is not enough
    def np_hash(self, data):
        return self.game.get_hash(data)

    # Run a MCTS simulation starting from state s of the tree.
    # The tree is accumulated in the self.tree dictionary.
    # The epsilon fix prevents the U term from being 0 when unexplored (N=0).
    # With the fix, priors (P) can be factored in immediately during selection and expansion.
    # This makes the search more efficient, given there are strong priors.
    def simulate(self, s, cpuct=1, epsilon_fix=True):
        hashed_s = self.np_hash(s) # Key for state in dictionary
        current_player = self.game.get_player(s)
        if hashed_s in self.tree: # Not at leaf; select.
            print("select env step: ", s["env"].step_cnt)
            stats = self.tree[hashed_s]
            N, Q, P = stats["obs"][:,MCTS_STOAGE_INDEX_N], stats["obs"][:,MCTS_STOAGE_INDEX_Q], stats["obs"][:,MCTS_STOAGE_INDEX_P]
            U = cpuct*P*math.sqrt(N.sum() + (1e-6 if epsilon_fix else 0))/(1 + N)
            heuristic = Q + U
            best_a_idx = np.argmax(heuristic)
            best_a = stats["obs"][best_a_idx, MCTS_STOAGE_INDEX_BESTACTION] # Pick best action to take
            if 0:#"child" in stats and best_a[0] in stats["child"]:
                print("Step was taken before: ", best_a)
                print("Child ", stats["child"])
                scores = self.simulate(self.tree[stats["child"][best_a[0]]])
            else:
                template = np.zeros_like(self.game.get_available_actions(s)) # Submit action to get s'
                template[tuple(best_a)] = True
                s_prime = self.game.take_action(s, template)
                hs_prime = self.np_hash(s_prime)

                # recursion is not always bad in all games...
                # maybe in your game you have to pull a new card from stack 
                # or roll a dice until you have a specific value...
                # action masking will consider for correct gameplay
                if hs_prime == hashed_s:
                    print("Recursion detected with action: ", best_a)
                    if hs_prime in self.tree:
                        print("Step was taken before: ", best_a)
                scores = self.simulate(s_prime) # Forward simulate with this action
            n, q = N[best_a_idx], Q[best_a_idx]
            v = scores[current_player] # Index in to find our reward
            stats["obs"][best_a_idx, MCTS_STOAGE_INDEX_Q] = (n*q+v)/(n + 1)
            stats["obs"][best_a_idx, MCTS_STOAGE_INDEX_N] += 1
            stats["child"] = {best_a[0]:hs_prime}
            return scores

        else: # Expand
            print("expand env step: ", s["env"].step_cnt)
            if self.game.check_game_over(s): # Reached a terminal node
                score = self.game.get_scores(s)
                return score
            available_actions = self.game.get_available_actions(s)
            idx = np.stack(np.where(available_actions)).T
            print("s", s["obs"].shape)
            p, v = self.nn.predict(s, available_actions)
            print("p", p.shape, "v",v.shape)
            ### TODO: settings to config file
            if self.add_noise:
                alpha = 0.03
                weight = 0.20
                noise = np.random.dirichlet([alpha]*np.atleast_1d(p).shape[0])
                p = p*(1.-weight) + noise*weight
            ###
            stats = np.zeros((len(idx), 4), dtype=np.object)
            stats[:,-1] = p
            stats[:,0] = list(idx)
            self.tree[hashed_s] = { "env":s["env"], "obs":stats }
            # if we have perfect information about the current game states, 
            # we could use this to speed up the training
            # TODO: add later a regulation paramter to consider more value head outputs
            # playing with scores from current game state only could result in simpler
            # strategies in a lot of games
            v_from_game = self.game.get_scores(s)
            return v_from_game if v_from_game is not None else v


    # Returns the MCTS policy distribution for state s.
    # The temperature parameter softens or hardens this distribution.
    def get_distribution(self, s, temperature):
        hashed_s = self.np_hash(s)
        stats = self.tree[hashed_s]["obs"][:,:2].copy()
        N = stats[:,1]
        try:
            raised = np.power(N, 1./temperature)
        # As temperature approaches 0, the effect becomes equivalent to argmax.
        except (ZeroDivisionError, OverflowError):
            raised = np.zeros_like(N)
            raised[N.argmax()] = 1
        
        total = raised.sum()
        # If all children are unexplored, prior is uniform.
        if total == 0:
            raised[:] = 1
            total = raised.sum()
        dist = raised/total
        stats[:,1] = dist
        return stats
        

