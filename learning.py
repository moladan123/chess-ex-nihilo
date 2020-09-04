# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

from copy import deepcopy
from tqdm import tqdm

import chess

WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
DRAW = "1/2-1/2"

def game_result(game_metadata: dict):
    result = game_metadata['Result']
    if result == WHITE_WIN:
        return 1
    elif result == BLACK_WIN:
        return -1
    else:
        return 0

def board_states(game_metadata: dict):
    s = chess.State()
    boards = []
    for move in game_metadata['moves'][:-1]:
        s.move(move)
        boards.append(deepcopy(s))
    return boards

def state_to_tensors(state: chess.State, result: int):
    pass


if __name__ == "__main__":
    all_games = chess.get_games_data("games/lichess_elite_2015-12.pgn")

    all_board_states = []

    fails = 0

    for game_metadata in tqdm(all_games):
        try:
            states = board_states(game_metadata)
        except ValueError:
            # In the case that we were not able to parse a move
            # so we assume we messed up somewhere and don't keep any of the states
            # This is ok if it is a small number of games
            states = []
            fails += 1
        except:
            pass

        result = game_result(game_metadata)

        for state in states:
            all_board_states.append((state, result))

    print("failed to parse " + str(fails) + "/" + str(len(all_games)) + " games")
    print(len(all_board_states))

