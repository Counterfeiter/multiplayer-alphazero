from itertools import permutations
import numpy as np

# Runs a match with the given game and list of players.
# Returns an array of points. Player number is the index into this array.
# For each match, a player gains a point if it wins, loses a point if it loses,
# and gains no points if it ties.
def play_match(game, players, verbose=False, permute=False):

    # You can use permutations to break the dependence on player order in measuring strength.
    matches = list(permutations(np.arange(len(players)))) if permute else [np.arange(len(players))]
    
    # Initialize scoreboard
    scores = np.zeros(len(players))

    # Run the matches (there will be multiple if permute=True)
    print("Playing " + str(matches) + "...")
    for order in matches:

        for p in players:
            p.reset() # Clear player trees to make the next match fair

        s = game.get_initial_state()
        if verbose: game.visualize(s)
        game_over = game.check_game_over(s)

        while not game_over:
            p = order[game.get_player(s)]
            if verbose: print("Player #{}'s turn.".format(p))
            s = players[p].update_state(s)
            if verbose: game.visualize(s)
            game_over = game.check_game_over(s)

        scores[list(order)] += game.get_scores(s)
        if verbose: print("Δ" + str(game.get_scores(s)) + ", Current scoreboard: " + str(scores))


    if verbose: print("Final scores:", scores)
    return scores


def play_load_match(game, players, loadfile, verbose=False):

    # Initialize scoreboard
    scores = np.zeros(len(players))

    s = game.get_initial_state(loadgamefile=loadfile, train = False)
    if verbose: game.visualize(s)
    game_over = game.check_game_over(s)

    while not game_over:
        p = game.get_player(s)
        if verbose: print("Player #{}'s turn.".format(p))
        s = players[p].update_state(s)
        if verbose: game.visualize(s)
        game_over = game.check_game_over(s)

    game.get_scores(s)
    if verbose: print("Δ" + str(game.get_scores(s)) + ", Current scoreboard: " + str(scores))


    if verbose: print("Final scores:", scores)
    return scores


if __name__ == "__main__":
    from players.human_player import HumanPlayer
    from neural_network import NeuralNetwork
    from models.senet import SENet
    from models.senet2 import SENet2
    from models.senetbig import SENetBig
    from models.ffnet import FFNet
    from models.senetmixed import SENetMixed
    from models.convnet import ConvNet
    from players.uninformed_mcts_player import UninformedMCTSPlayer
    from players.deep_mcts_player import DeepMCTSPlayer
    from games.tictactoe import TicTacToe
    from games.tictacmo import TicTacMo
    from games.kingdombuilder import AZKingdomBuilder as KindgomBuilder


    # Change these variable 
    game = KindgomBuilder()
    ckpt = 30
    nn = NeuralNetwork(game, FFNet, cuda=True)
    nn.load(ckpt)
    
    # HumanPlayer(game),
    # UninformedMCTSPlayer(game, simulations=1000)
    opponents = [HumanPlayer(game)]
    #opponents = [DeepMCTSPlayer(game, nn, simulations=2000)]
    #opponents = [UninformedMCTSPlayer(game, simulations=500) for _ in range(1)]
    ai = DeepMCTSPlayer(game, nn, simulations=2000)
    #ai = UninformedMCTSPlayer(game, simulations=10000)
    
    players = [ai] + opponents
    play_load_match(game, players, "initalgame.ini", verbose=True)
    
