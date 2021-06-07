import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from copy import deepcopy
import numpy as np

import random
from enum import Enum, unique

from ray.rllib.env.multi_agent_env import MultiAgentEnv


from kingdombuilder import Game, DOACTION, quadrants
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

def obs_hex_axial(obs):
    cnn_in = obs["obs"]["cnn_input"]
    cnn_size = cnn_in.shape
    new_cnn = np.zeros( (cnn_size[0], cnn_size[1] * 2 - 1, cnn_size[2]), dtype=np.float32 )

    row_cnt = 0
    for col_offset in range(cnn_size[1] - 1, 0, -1):
        for _ in range(2):
            if row_cnt >= cnn_size[0]:
                break
            new_cnn[row_cnt,col_offset:col_offset + cnn_size[1],:] = cnn_in[row_cnt,:,:]
            row_cnt += 1

    obs["obs"]["cnn_input"] = new_cnn

    return obs


class ParametricCNNKingdomBuilderEnv(gym.Env):
    max_avail_actions = 2802 + 5 * 400
    #max_avail_actions = len(DOACTION) + 20*20

    max_player = 5 # player settlements if placed = 1.0
    # kingdom builder is a game where each settlements part of the reward or gold could be calulated
    # the next step could be to weight this gold earning houses in the cnn layer
    players_turn = max_player # 0.5 if player is in the game and 1.0 if its players turn (wholes plain)

    # informations fixed... could be calc if game starts
    terrains = len(TERRAINANDSPECIAL) # each terrain gets his own channel
    current_card = len(TERRAIN)
    towns = len(BOARDSECTIONS)
    rules = len(CARDRULES) # cardrules one per layer, if rule is used, set whole layer to one

    CNN_layer = terrains + players_turn + rules + towns

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
        towns_list = [SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNEMPTY.value]
        obs = np.zeros(shape=(20, 20, self.rules + self.towns), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                if self.game.board.board_env[row][col] in towns_list:
                    idx = self._town_to_boardsection(row, col).value
                    obs[row,col,self.rules + idx] = 1.0

        for rule in self.game.rules.rules:
            obs[:,:,rule.value] = 1.0

        return obs

    def _extract_observations(self):
        obs = np.zeros(shape=(20, 20, self.terrains + self.players_turn), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                #terrains
                if self.game.board.board_env[row][col] in [e.name for e in TERRAINANDSPECIAL]:
                    if self.game.board.board_env[row][col] == self.game.player.card.value:
                        obs[row,col,TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value] = 0.5
                    else:
                        obs[row,col,TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value] = 1.0

                # check player settlements and turn indication
                player_settle = int(self.game.board.board_settlements[row][col])
                if player_settle != 0:
                    if player_settle == int(self.game.player):
                        obs[row, col, player_settle - 1] = 1.0 # player has settlements and players turn
                    else:
                        obs[row, col, player_settle - 1] = 0.5 # player has settlements 
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

    def step(self, action, force_card = None):
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

        if force_card is not None:
            self.oldplayer.current_card = force_card

        if self.game.done:
            #print("Game done")
            self.done = True

        self.calcmask()

        self.dynamic_obs = self._extract_observations()

        dict_obs = {
            "action_mask": self.action_mask,
            "obs": {
                "cnn_input":np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2),
                "ff_input":np.array([], dtype=np.float32) #dummy
            }
        }

        dict_obs = obs_hex_axial(dict_obs)

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
                self.game = Game(2, rules=fixed_rules, deterministic = self.train_mode)
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
            "obs": {
                "cnn_input":np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2),
                "ff_input":np.array([], dtype=np.float32) #dummy
            }
        }

        dict_obs = obs_hex_axial(dict_obs)

        return dict_obs

    def render(self, mode='human', close=False):
        if mode == 'human':
            print("Player {:s} with Terrain {:s} and towns to play: ".format(str(self.game.player), self.game.player.card.name), self.game.townstoplay)
            print(self.game)


class ParametricMixedKingdomBuilderEnv(gym.Env):
    max_avail_actions = 2802 + 5 * 400
    #max_avail_actions = len(DOACTION) + 20*20

    max_player = 5 # player settlements if placed = 1.0
    # kingdom builder is a game where each settlements part of the reward or gold could be calulated
    # the next step could be to weight this gold earning houses in the cnn layer
    players_turn = max_player # 0.5 if player is in the game and 1.0 if its players turn (wholes plain)

    # informations fixed... could be calc if game starts
    terrains = len(TERRAINANDSPECIAL) # each terrain gets his own channel
    current_card = len(TERRAIN)
    towns = len(BOARDSECTIONS)
    rules = len(CARDRULES) # cardrules one per layer, if rule is used, set whole layer to one

    CNN_layer = terrains + players_turn + towns

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

        self.towns_list = [ x for x in BOARDSECTIONS.list()]
        self.towns_list += self.towns_list
        self.towns_list.remove(BOARDSECTIONS.ORACLE)
        self.towns_list.remove(BOARDSECTIONS.HARBOR)

        self.reset()

    def _town_to_boardsection(self, row, col):
        if self.game.board.board_env[row][col] not in [SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNEMPTY.value]:
            return None
        ind = (row // 10) * 2  + (col // 10)
        return BOARDSECTIONS[self.game.board.quadrant_order[ind]]

    def _extract_fixedobservations(self):
        towns_list = [SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNEMPTY.value]
        obs = np.zeros(shape=(20, 20, self.towns), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                if self.game.board.board_env[row][col] in towns_list:
                    idx = self._town_to_boardsection(row, col).value
                    obs[row,col,idx] = 1.0

        return obs

    def _extract_observations(self):
        obs = np.zeros(shape=(20, 20, self.terrains + self.players_turn), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                #terrains
                if self.game.board.board_env[row][col] in [e.name for e in TERRAINANDSPECIAL]:
                    if self.game.board.board_env[row][col] == self.game.player.card.value:
                        obs[row,col,TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value] = 0.5
                    else:
                        obs[row,col,TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value] = 1.0

                # check player settlements and turn indication
                player_settle = int(self.game.board.board_settlements[row][col])
                if player_settle != 0:
                    if player_settle == int(self.game.player):
                        obs[row, col, player_settle - 1] = 1.0 # player has settlements and players turn
                    else:
                        obs[row, col, player_settle - 1] = 0.5 # player has settlements 
        return obs

    def _extract_extraobservations(self):
        values = []
        for i in range(5):
            if i < len(self.game.players):
                player = self.game.players[i]
                values.append(1.0) # player in game
                values.append(player.settlements / 40.0)
                values.append(1.0 if self.game.current_player == i else 0.0)
            else:
                values.append(0.0) # player in game
                values.append(0.0)
                values.append(0.0)

        for rule in CARDRULES.list():
            if rule in self.game.rules.rules:
                values.append(1.0)
            else:
                values.append(0.0)

        towns_current = self.game.townstoplay.copy()
        for town in self.towns_list:
            if town in towns_current:
                values.append(1.0)
                towns_current.remove(town)
            else:
                values.append(0.0)

        for action in list(DOACTION):
            if self.game.old_action == action:
                values.append(1.0)
            else:
                values.append(0.0)

        return values

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

    def unravel_boardactions(self, board_action):
        actions = list(np.unravel_index(board_action, (20, 20)))
        return actions

    def step(self, action, force_card = None):
        #print("Step action", action)
        action_board = self.unravel_actions(action)

        self.step_cnt += 1

        reward = 0

        #print(actions)
        self.oldplayer = self.game.player
        if not self.game.singlestepmove(DOACTION(action_board[0]), action_board[1], action_board[2]):
            #could not happen in parametric action spaces
            self.done = True
            reward = -1
            print("Game ends with invalid action")
        else:
            self.last_action = DOACTION(action_board[0])

        if force_card is not None:
            self.oldplayer.current_card = force_card

        if self.game.done:
            #print("Game done")
            self.done = True

        self.calcmask()

        self.dynamic_obs = self._extract_observations()

        dict_obs = {
            "action_mask": self.action_mask,
            #"avail_actions": self.avail_actions,
            "obs": {
                "cnn_input":np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2),
                "ff_input":np.array(self._extract_extraobservations(), dtype=np.float32)
            }
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
                self.game = Game(2, deterministic = self.train_mode)
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
            "obs": {
                "cnn_input":np.concatenate( (self.dynamic_obs, self.fixed_obs), axis=2),
                "ff_input":np.array(self._extract_extraobservations(), dtype=np.float32)
            }
        }

        return dict_obs

    def render(self, mode='human', close=False):
        if mode == 'human':
            print("Player {:s} with Terrain {:s} and towns to play: ".format(str(self.game.player), self.game.player.card.name), self.game.townstoplay)
            if self.game.main_move > 0:
                moves = self.game.board.getpossiblemove(self.game.player, self.game.player.card)
                print(self.game.board.print_selection(moves))
            else:
                print(self.game)

class ParametricFFKingdomBuilderEnv(gym.Env):
    max_avail_actions = 2802 + 5 * 400
    #max_avail_actions = len(DOACTION) + 20*20

    max_player = 5 # player settlements if placed = 1.0
    # kingdom builder is a game where each settlements part of the reward or gold could be calulated
    # the next step could be to weight this gold earning houses in the cnn layer
    players_turn = max_player # 0.5 if player is in the game and 1.0 if its players turn (wholes plain)

    # informations fixed... could be calc if game starts
    terrains = len(TERRAINANDSPECIAL) # each terrain gets his own channel
    current_card = len(TERRAIN)
    towns = len(BOARDSECTIONS)
    rules = len(CARDRULES) # cardrules one per layer, if rule is used, set whole layer to one

    CNN_layer = terrains + players_turn + towns

    action_space=Discrete(max_avail_actions)

    def __init__(self, level='easy', normalize_obs = False, verbose=1, fail_actions = 1, train = False):

        self.avail_actions = np.ones(self.max_avail_actions, dtype=np.float32).flatten()
        self.train_mode = train
        self.done = False
        self.level = level
        self.verbose = verbose

        self.towns_list = [ x for x in BOARDSECTIONS.list()]
        self.towns_list += self.towns_list
        self.towns_list.remove(BOARDSECTIONS.ORACLE)
        self.towns_list.remove(BOARDSECTIONS.HARBOR)

        self.reset()

    def _town_to_boardsection(self, row, col):
        if self.game.board.board_env[row][col] not in [SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNEMPTY.value]:
            return None
        ind = (row // 10) * 2  + (col // 10)
        return BOARDSECTIONS[self.game.board.quadrant_order[ind]], ind

    def _extract_fixedobservations(self):
        towns_list = [SPECIALLOCATION.TOWNFULL.value, SPECIALLOCATION.TOWNHALF.value, SPECIALLOCATION.TOWNEMPTY.value]

        current_town_list = [0.0] * 8
        for row in range(20):
            for col in range(20):
                if self.game.board.board_env[row][col] in towns_list:
                    town_type, idx = self._town_to_boardsection(row, col)
                    townvalue = 0.2 + town_type.value / (len(self.game.board.env_quadrants["quadrants"]) * 1.25)
                    current_town_list[idx*2] = townvalue
                    current_town_list[idx*2 + 1] = townvalue

        return np.array(current_town_list, dtype=np.float32)

    def _extract_observations(self):
        obs = np.zeros(shape=(20, 20, 2), dtype=np.float32)
        for row in range(20):
            for col in range(20):
                #terrains
                obs[row,col,0] = TERRAINANDSPECIAL[self.game.board.board_env[row][col]].value / (len(TERRAINANDSPECIAL) - 1)

                # check player settlements
                player_settle = int(self.game.board.board_settlements[row][col])
                if player_settle != 0:
                    obs[row, col, 1] = player_settle / 5.0

        return obs

    def _extract_extraobservations(self):
        values = []
        for i in range(5):
            if i < len(self.game.players):
                player = self.game.players[i]
                values.append(1.0) # player in game
                values.append(player.settlements / 40.0)
                values.append(1.0 if self.game.current_player == i else 0.0)
            else:
                values.append(0.0) # player in game
                values.append(0.0)
                values.append(0.0)

        for rule in CARDRULES.list():
            if rule in self.game.rules.rules:
                values.append(1.0)
            else:
                values.append(0.0)

        towns_current = self.game.townstoplay.copy()
        for town in self.towns_list:
            if town in towns_current:
                values.append(1.0)
                towns_current.remove(town)
            else:
                values.append(0.0)

        for action in list(DOACTION):
            if self.game.old_action == action:
                values.append(1.0)
            else:
                values.append(0.0)

        return np.array(values, dtype=np.float32)

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

    def unravel_boardactions(self, board_action):
        actions = list(np.unravel_index(board_action, (20, 20)))
        return actions

    def step(self, action, force_card = None):
        #print("Step action", action)
        action_board = self.unravel_actions(action)

        self.step_cnt += 1

        reward = 0

        #print(actions)
        self.oldplayer = self.game.player
        if not self.game.singlestepmove(DOACTION(action_board[0]), action_board[1], action_board[2]):
            #could not happen in parametric action spaces
            self.done = True
            reward = -1
            print("Game ends with invalid action")
        else:
            self.last_action = DOACTION(action_board[0])

        if force_card is not None:
            self.oldplayer.current_card = force_card

        if self.game.done:
            #print("Game done")
            self.done = True

        self.calcmask()

        self.dynamic_obs = self._extract_observations()
        self.extra = self._extract_extraobservations()

        ff_in = np.concatenate((self.fixed_obs.flatten(), self.dynamic_obs.flatten(), self.extra))

        dict_obs = {
            "action_mask": self.action_mask,
            #"avail_actions": self.avail_actions,
            "obs": {
                "cnn_input":np.array([]),
                "ff_input":ff_in
            }
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
                self.game = Game(2, deterministic = self.train_mode)
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
        self.extra = self._extract_extraobservations()

        ff_in = np.concatenate((self.fixed_obs.flatten(), self.dynamic_obs.flatten(), self.extra))

        dict_obs = {
            "action_mask": self.action_mask,
            "obs": {
                "cnn_input":np.array( [] ),
                "ff_input":ff_in
            }
        }

        return dict_obs

    def render(self, mode='human', close=False):
        if mode == 'human':
            print("Player {:s} with Terrain {:s} and towns to play: ".format(str(self.game.player), self.game.player.card.name), self.game.townstoplay)
            if self.game.main_move > 0:
                moves = self.game.board.getpossiblemove(self.game.player, self.game.player.card)
                print(self.game.board.print_selection(moves))
            else:
                print(self.game)