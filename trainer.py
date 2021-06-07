import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from mcts import MCTS
from mcts_ray import MCTS as MCTSRAY, Node, RootParentNode
from play import play_match
from players.uninformed_mcts_player import UninformedMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer
from kingdombuilder import CARDRULES

# Object that coordinates AlphaZero training.
class Trainer:

    def __init__(self, game, nn, num_simulations, num_games, num_updates, buffer_size_limit, cpuct, num_threads, writer):
        self.game = game
        self.nn = nn
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.num_updates = num_updates
        self.buffer_size_limit = buffer_size_limit
        self.training_data = np.zeros((0,4))
        self.cpuct = cpuct
        self.num_threads = num_threads
        self.error_log = []
        self.writer = writer

        self.games_cnt = 0


    # Does one game of self play and generates training samples.
    def self_play(self, temperature, nn):
        s = self.game.get_initial_state()
        tree = MCTS(self.game, nn, add_noise=True)

        data = []
        done = self.game.check_game_over(s)

        while not done:
            
            # Think
            for _ in range(self.num_simulations):
                tree.simulate(s, cpuct=self.cpuct)

            # Fetch action distribution and append training example template.
            dist = tree.get_distribution(s, temperature=temperature)

            available = self.game.get_available_actions(s)

            data.append([s["obs"], dist[:,1], None, available]) # state, prob, outcome, action_mask

            # Sample an action
            idx = np.random.choice(len(dist), p=dist[:,1].astype(np.float))
            a = tuple(dist[idx, 0])

            # Apply action
            template = np.zeros_like(available)
            template[a] = 1
            s = self.game.take_action(s, template)

            # Check scores
            done = self.game.check_game_over(s)

        scores = self.game.get_scores(s)

        # Update training examples with outcome
        for i, _ in enumerate(data):
            data[i][2] = scores

        print("Tree length", len(tree.tree))

        return np.array(data, dtype=np.object)

    def self_play_ray(self, temperature, nn):
        mcts_config = {
            "puct_coefficient": 0.7,
            "num_simulations": self.num_simulations,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.25,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": False,
        }
        s = self.game.get_initial_state()
        tree = MCTSRAY(nn, mcts_config)

        done = self.game.check_game_over(s)

        tree_node = Node(
                        state=s,
                        reward=0,
                        done=done,
                        action=None,
                        parent=RootParentNode(env=self.game, state=s),
                        mcts=tree)

        data = []

        explored_tree = 0.0
        steps = 0
        
        while not done:
            
            available = self.game.get_available_actions(s)

            # Think
            p, action, new_tree_node = tree.compute_action(tree_node)

            # free memory
            #for nodes in tree_node.children:
            #    if new_tree_node != nodes:
            #        del nodes

            #tree_node = new_tree_node

            p = nn.get_valid_dist(p, available).cpu().numpy().squeeze()

            data.append([s["obs"], p, None, available]) # state, prob, outcome, action_mask

            # Apply action
            template = np.zeros_like(available)
            template[action] = 1
            s_new = self.game.take_action(s, template)
            match = False
            for node in new_tree_node:
                if s_new["env"].game.gamestate_to_dict() == node.state["env"].game.gamestate_to_dict():
                    tree_node = node
                    match = True
                    break

            explored_tree += len(new_tree_node[0].children) / np.count_nonzero(available)
            steps += 1
            
            if match != True:
                print("debug")
            assert match, "hash does not match, games does not follow MCTS path"
            s = s_new

            # Check scores
            done = self.game.check_game_over(s)

        #game done
        self.games_cnt += 1

        self.writer.add_scalar('MCTS/Exploring', explored_tree / steps, self.games_cnt)
        
        scores = self.game.get_scores(s)

        if self.writer is not None:
            for card in CARDRULES.list():
                if card in s["env"].game.rules.rules:
                    playersum = 0
                    for player in s["env"].game.players:
                        playersum += s["env"].game.rules.player_score_per_rule[int(player) - 1][card.value]
                    self.writer.add_scalar('Score/' + card.name, playersum / len(s["env"].game.players), self.games_cnt)

        scores = np.append(scores, self.game.get_current_players(s))

        # Update training examples with outcome
        for i, _ in enumerate(data):
            data[i][2] = scores

        return np.array(data, dtype=np.object)

    # Performs one iteration of policy improvement.
    # Creates some number of games, then updates network parameters some number of times from that training data.
    def policy_iteration(self, verbose=False):
        temperature = 3.0

        if verbose:
            print("SIMULATING " + str(self.num_games) + " games")
            start = time.time()
        if self.num_threads > 1:
            jobs = [(temperature, self.nn)]*self.num_games
            pool = ThreadPool(self.num_threads)
            new_data = pool.map(self.self_play_ray, jobs)
            pool.close()
            pool.join()
            self.training_data = np.concatenate([self.training_data] + new_data, axis=0)
        else:
            for _ in range(self.num_games): # Self-play games
                #new_data = self.self_play(temperature, self.nn)
                new_data = self.self_play_ray(temperature, self.nn)
                self.training_data = np.concatenate([self.training_data, new_data], axis=0)
        if verbose:
            print("Simulating took " + str(int(time.time()-start)) + " seconds")

        if verbose:
            print("TRAINING")
            start = time.time()
        mean_loss = None
        count = 0
        for _ in range(self.num_updates):
            self.nn.train(self.training_data)
            new_loss = self.nn.latest_loss.item()
            if mean_loss is None:
                mean_loss = new_loss
            else:
                (mean_loss*count + new_loss)/(count+1)
            count += 1
        self.error_log.append(mean_loss)

        if verbose:
            print("Training took " + str(int(time.time()-start)) + " seconds")
            print("Average train error:", mean_loss)

        # Prune oldest training samples if a buffer size limit is set.
        if self.buffer_size_limit is not None:
            self.training_data = self.training_data[-self.buffer_size_limit:,:]

