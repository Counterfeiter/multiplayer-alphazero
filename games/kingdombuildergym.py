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

class ParametricCNNKingdomBuilderEnv(gym.Env):
    max_avail_actions = 2802 + 5 * 400

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

        self.avail_actions = np.ones(self.max_avail_actions, dtype=np.float32).flatten()
        self.train_mode = train
        self.done = False
        self.level = level
        self.verbose = verbose
        self.reset()

    def _town_to_boardsection(self, row, col):
        if self.game.board.board_env[row][col] not in [SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNEMPTY.value]:
            return None
        ind = (row // 10) * 2  + (col // 10)
        return BOARDSECTIONS[self.game.board.quadrant_order[ind]]

    def _extract_fixedobservations(self):
        obs = np.zeros(shape=(20, 20, self.terrains + self.rules + self.towns), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                if self.game.board.board_env[row][col] in [e.name for e in TERRAINANDSPECIAL]:
                    obs[row,col,TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value] = 1.0
                towns_list = [SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNEMPTY.value]
                if self.game.board.board_env[row][col] in towns_list:
                    idx = self._town_to_boardsection(row, col).value
                    obs[row,col,self.terrains + self.rules + idx] = 1.0

        for rule in self.game.rules.rules:
            obs[:,:,self.terrains + rule.value] = 1.0

        return obs

    def _extract_observations(self):
        obs = np.zeros(shape=(20, 20, self.max_player + self.players_turn), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                if int(self.game.board.board_settlements[row][col]) != 0:
                    obs[row, col, int(self.game.board.board_settlements[row][col]) - 1] = 1.0
            
        for player in self.game.players:
            if player == self.game.player:
                obs[:,:,self.max_player + int(player) - 1] = 1.0
            else:
                obs[:,:,self.max_player + int(player) - 1] = 0.5
            
        return obs

    def calcmask(self):
        pa = self.game.actionstomoves()
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

        reward = 0

        #print(actions)
        self.oldplayer = self.game.player
        if not self.game.singlestepmove(DOACTION(actions[0]), actions[1], actions[2]):
            #could not happen in parametric action spaces
            self.done = True
            reward = -1
            print("Game ends with invalid action")
        else:
            self.last_action = DOACTION(actions[0])
            #self.game.rules.score(self.game.players)
            #reward = self.game.player.score - self.old_player_scores[int(self.game.player) - 1]
            #self.old_player_scores[int(self.game.player) - 1] = self.game.player.score

            #if self.verbose:
            #    print("Action {:s} succeed with reward {:d}".format(DOACTION(actions[0]).name, reward))

        if self.game.done:
            #print("Game done")
            self.done = True

        self.calcmask()

        self.dynamic_obs = self._extract_observations()

        dict_obs = {
            "action_mask": self.action_mask,
            #"avail_actions": self.avail_actions,
            "obs": np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2)
        }

        return dict_obs, reward, self.done, {}

    def reset(self, load_game : str = ""):

        if load_game != "":
            self.game = Game.load(load_game)
        else:
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
                self.game = Game(3, fixed_quadrants, deterministic = self.train_mode)
            elif self.level == 'professional':
                #self.game = Game(4, deterministic = self.train_mode)
                self.game = Game(random.randint(2, 5), deterministic = self.train_mode)
            elif self.level == 'professional2player':
                self.game = Game(2, deterministic = self.train_mode)
            else:
                assert(False)

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

        return dict_obs

    def render(self, mode='human', close=False):
        if mode == 'human':
            print("Player {:s} with Terrain {:s} and towns to play: ".format(str(self.game.player), self.game.player.card.name), self.game.townstoplay)
            print(self.game)