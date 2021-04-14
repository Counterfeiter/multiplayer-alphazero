import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from copy import deepcopy
import numpy as np

import random
from enum import Enum, unique

from ray.rllib.env.multi_agent_env import MultiAgentEnv


from kingdombuilder import Game, DOACTION
from kingdombuilder import Board
from kingdombuilder import Rules
from kingdombuilder import Player
from kingdombuilder import CARDRULES, TERRAIN, SPECIALLOCATION, BOARDSECTIONS

@unique
class TERRAINANDSPECIAL(Enum):
    G = 0
    B = 1
    F = 2
    S = 3
    D = 4
    W = 5
    M = 6
    C = 7
    T = 8
    t = 9
    X = 10

@unique
class TOWNSTOPLAY(Enum):
    ORACLE = 0
    HARBOR = 1
    FARM_1 = 2
    FARM_2 = 3
    PADDOCK_1 = 4
    PADDOCK_2 = 5


scalerback = lambda x, xmin, xmax : x * (xmax - xmin) + xmin
scalernorm = lambda x, xmin, xmax : (x - xmin) / (xmax - xmin)

class KingdomBuilderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    obser_min = np.array([
                            *([0] * 20*20),
                            *([0] * 20*20),
                            *([0] * len(DOACTION)),
                            *([0] * 5), #current player
                            *([0] * len(TOWNSTOPLAY)),
                            *([0] * len(TERRAIN)),
                            *([0]), # all settlements left
                            *([0]), # settlements left main move 0 - 3
                            *([0] * len(CARDRULES)),
                        ], dtype=np.float32)

    obser_max = np.array([
                            *([len(TERRAINANDSPECIAL)] * 20*20),
                            *([5] * 20*20),
                            *([1] * len(DOACTION)),
                            *([1] * 5), #current player
                            *([1] * len(TOWNSTOPLAY)),
                            *([1] * len(TERRAIN)),
                            *([Player.MAX_SETTLEMENTS]),
                            *([3]), # settlements left main move 0 - 3
                            *([1] * len(CARDRULES)),
                        ], dtype=np.float32)

    observation_space=Box(low=obser_min, high=obser_max, dtype=np.float32)

    def __init__(self, level='easy', normalize_obs = True, verbose=1, fail_actions = 1, train = False):
        """
        Observation space:
            #Start with a minimum information set and extend later
            - board env 20x20 -> scaled TERRAIN + SPECIALLOCATION = TERRAINANDSPECIAL
            - board settlements 20x20 -> scaled 1 - 5 player settlement
            - current action: main move, town harbor 1 select, town paddock select -> bool
            - current player 5x -> bool
            - current towns left to play: town oracle 1x, town harbor 1x, town farm 2x, town paddock 2x -> bool
            - current terrain card len(TERRAIN) -> bool
            - current settlements left -> scaled 0 - 40 settlements
            - rule cards len(CARDRULES) -> bool
        """
        self.normalize_obs = normalize_obs



        if self.normalize_obs:
            scaled_min = scalernorm(self.obser_min, self.obser_min, self.obser_max)
            scaled_max = scalernorm(self.obser_max, self.obser_min, self.obser_max)
        else:
            scaled_min = self.obser_min
            scaled_max = self.obser_max

        """
        Action space:

            1) actions
            2) row number - 20
            3) col number - 20
        """
        self.action_space=MultiDiscrete([len(DOACTION), 20, 20])

        self.level = level
        self.verbose = verbose
        assert(fail_actions > 0)
        self.fail_actions = fail_actions
        self.train_mode = train
        self.state = None

        self.reset()

    def _extract_observations(self):
        obs = np.array(([0] * 2*20*20))
        for i in range(20*20):
            row = i % 20
            col = i // 20
            obs[i] = TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value
            obs[i + 20*20] = int(self.game.board.board_settlements[row][col])
        act = [0] * len(DOACTION)
        act[self.last_action.value] = 1
        obs = np.append(obs, act)

        player = [0] * 5
        player[int(self.game.player) - 1] = 1
        obs = np.append(obs, player)

        towns_to_play = []
        towns = self.game.player.towns.copy()
        for town in TOWNSTOPLAY:
            if BOARDSECTIONS[town.name.split('_')[0]] in towns:
                towns_to_play.append(1)
                towns.remove(BOARDSECTIONS[town.name.split('_')[0]])
            else:
                towns_to_play.append(0)
        obs = np.append(obs, towns_to_play)

        terrain = [0] * len(TERRAIN)
        for i, ter in enumerate(TERRAIN):
            if self.game.player.card == ter:
                terrain[i] = 1
        obs = np.append(obs, terrain)

        obs = np.append(obs, self.game.player.settlements)
        obs = np.append(obs, self.game.main_move)

        obs = np.append(obs, self.cardrules)

        assert(len(self.obser_min) == len(obs))

        return obs.astype(np.float32)


    def step(self, actions):
        reward = 0
        done = False

        self.oldplayer = self.game.player
        if not self.game.singlestepmove(DOACTION(actions[0]), actions[1], actions[2]):
            #could not happen in parametric action spaces
            self.fail_counter += 1
            reward = -(self.fail_counter * (1.0 / self.fail_actions))
            if self.fail_counter >= self.fail_actions:
                done = True
        else:
            self.last_action = DOACTION(actions[0])
            self.game.rules.score(self.game.players)
            reward = self.game.player.score - self.old_player_scores[int(self.game.player) - 1]
            self.old_player_scores[int(self.game.player) - 1] = self.game.player.score

            if self.verbose:
                print("Action {:s} succeed with reward {:d}".format(DOACTION(actions[0]).name, reward))

        if self.game.done:
            done = True
            #print(self.game.rules.player_score_per_rule)

        if self.normalize_obs:
            obs = scalernorm(self._extract_observations(), self.obser_min, self.obser_max)
        else:
            obs = self._extract_observations()

        return obs, reward, done, {}

    def reset(self):
        self.fail_counter = 0

        if self.level == 'easy':
            fixed_quadrants = ["ORACLE", "PADDOCK", "HARBOR", "FARM"]
            fixed_rules = [CARDRULES.MINERS, CARDRULES.FISHERMEN, CARDRULES.WORKER]
            fixed_rot = [False] * 4
            #only two players
            self.game = Game(2, fixed_quadrants, fixed_rot, fixed_rules, deterministic = self.train_mode)
        elif self.level == 'intermediate':
            fixed_rules = [CARDRULES.MINERS, CARDRULES.FISHERMEN, CARDRULES.WORKER]
            self.game = Game(4, deterministic = self.train_mode)
        elif self.level == 'advanced':
            fixed_quadrants = ["ORACLE", "PADDOCK", "HARBOR", "FARM"]
            #only two players
            self.game = Game(3, fixed_quadrants, deterministic = self.train_mode)
        elif self.level == 'professional':
            self.game = Game(4, deterministic = self.train_mode)
            #self.game = Game(random.randrange(1, 5), deterministic = self.train_mode)
        else:
            assert(False)

        self.last_action = DOACTION.END
        self.cardrules = [0] * len(CARDRULES)
        self.old_player_scores = [0] * len(self.game.players)
        for i, card in enumerate(CARDRULES):
            if card in self.game.rules.rules:
                self.cardrules[i] = 1

        if self.normalize_obs:
            obs = scalernorm(self._extract_observations(), self.obser_min, self.obser_max)
        else:
            obs = self._extract_observations()

        return obs

    def render(self, mode='human', close=False):
        if mode == 'human':
            print("Player {:s} with Terrain {:s} and towns to play: ".format(str(self.game.player), self.game.player.card.name), self.game.townstoplay)
            print("\nRule Cards: {:^16}{:^16}{:^16}{:^16}\n".format(*[x.name for x in self.game.rules.rules]))
            print(self.game.board)


class ParametricCNNKingdomBuilderEnv(gym.Env):
    max_avail_actions = 2802

    max_player = 5 # player settlements if placed = 1.0
    # kingdom builder is a game where each settlements part of the reward or gold could be calulated
    # the next step could be to weight this gold earning houses in the cnn layer
    players_turn = max_player # 0.5 if player is in the game and 1.0 if its players turn (wholes plain)

    # informations fixed... could be calc if game starts
    terrains = len(TERRAINANDSPECIAL) # each terrain gets his own channel
    towns = len(BOARDSECTIONS)
    rules = len(CARDRULES) # cardrules one per layer, if rule is used, set whole layer to one

    CNN_layer = max_player + players_turn + terrains + rules + towns

    observation_space = Dict({
        "action_mask": Box(0, 1, shape=(max_avail_actions, )),
        #"avail_actions": Box(-1, 1, shape=(max_avail_actions, CNN_layer)),
        "obs": Box(0, 1, shape=(CNN_layer, 20, 20)),
    })

    action_space=Discrete(max_avail_actions)

    def __init__(self, level='easy', normalize_obs = False, verbose=1, fail_actions = 1, train = False):

        self.wrapped = KingdomBuilderEnv(level=level, normalize_obs = normalize_obs, verbose=verbose, fail_actions = fail_actions, train = train)


        self.avail_actions = np.ones(self.max_avail_actions, dtype=np.float32).flatten()
        self.state = None
        self.reset()

    def _extract_fixedobservations(self):
        obs = np.zeros(shape=(20, 20, self.terrains + self.rules + self.towns), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                if self.wrapped.game.board.board_env[row][col] in [e.name for e in TERRAINANDSPECIAL]:
                    obs[row,col,TERRAINANDSPECIAL[self.wrapped.game.board.board_env[row][col]].value] = 1.0
                towns_list = [SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNEMPTY.value]
                if self.wrapped.game.board.board_env[row][col] in towns_list:
                    idx = self.wrapped.game.board.town_to_boardsection(row, col).value
                    obs[row,col,self.terrains + self.rules + idx] = 1.0

        for rule in self.wrapped.game.rules.rules:
            obs[:,:,self.terrains + rule.value] = 1.0

        return obs

    def _extract_observations(self):
        obs = np.zeros(shape=(20, 20, self.max_player + self.players_turn), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                if int(self.wrapped.game.board.board_settlements[row][col]) != 0:
                    obs[row, col, int(self.wrapped.game.board.board_settlements[row][col]) - 1] = 1.0
            
        for player in self.wrapped.game.players:
            if player == self.wrapped.game.player:
                obs[:,:,self.max_player + int(player) - 1] = 1.0
            else:
                obs[:,:,self.max_player + int(player) - 1] = 0.5
            
        return obs

    def calcmask(self):
        pa = self.wrapped.game.actionstomoves()
        action_mask = np.array([], dtype=np.bool)
        for key, value in pa.items():
            action_mask = np.append(action_mask, np.array(value, dtype=np.bool)) # append data

        if np.max(action_mask) < 1.0:
            print("No options to play")
            assert(False)

        self.action_mask = action_mask

    def unravel_actions(self, action):
        if action == 0:
            actions = np.array([0,0,0])
        elif action == 1:
            actions = np.array([1,0,0])
        else:
            actions = list(np.unravel_index(action - 2, (len(DOACTION) - 2, 20, 20)))
            actions[0] += 2

        return actions

    def action_to_string(self, action):
        actions = self.unravel_actions(action)
        return str(DOACTION(actions[0]))

    def step(self, action):
        #print("Step action", action)
        actions = self.unravel_actions(action)

        self.step_cnt += 1

        #print(actions)
        _, reward, self.done, info = self.wrapped.step(actions)
        self.calcmask()
        self.running_reward += reward

        self.dynamic_obs = self._extract_observations()

        dict_obs = {
            "action_mask": self.action_mask,
            #"avail_actions": self.avail_actions,
            "obs": np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2)
        }
        #dict_obs["obs"] = np.swapaxes(dict_obs["obs"],0,2)
        self.wrapped.state = dict_obs
        return dict_obs, self.running_reward if self.done else 0, self.done, info

    def reset(self):
        self.wrapped.reset()
        self.calcmask()
        self.done = False
        self.running_reward = 0.0

        self.step_cnt = 0

        self.fixed_obs = self._extract_fixedobservations()
        self.dynamic_obs = self._extract_observations()

        dict_obs = {
            "action_mask": self.action_mask,
            #"avail_actions": self.avail_actions,
            "obs": np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2)
        }
        #dict_obs["obs"] = np.swapaxes(dict_obs["obs"],0,2)
        self.wrapped.state = dict_obs
        return dict_obs

    def render(self, mode='human', close=False):
        self.wrapped.render(mode='human', close=False)

    def set_state(self, state):
        self.running_reward = state[1]
        self.wrapped = deepcopy(state[0])
        obs = self.wrapped.unwrapped.state
        return {"obs": obs["obs"], "action_mask": obs["action_mask"]}

    def get_state(self):
        return deepcopy(self.wrapped), self.running_reward