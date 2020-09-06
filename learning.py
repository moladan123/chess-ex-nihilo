try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Tensorflow is not installed")

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
    else:
        return 0

def board_states(game_metadata: dict):
    s = chess.State()
    boards = []
    for move in game_metadata['moves'][:-1]:
        s.move(move)
        boards.append(deepcopy(s))
    return boards

def state_to_int_list(state: chess.State):
    state_vars = []

    # One hot encoding of all 64 squares of the board
    for row in state.board:
        for square in row:
            for piece_type in "prnbqk PRNBQK":
                state_vars.append(1 if square == piece_type else 0)

    # Add which turn it is right now
    state_vars.append(1 if state.white_to_move else -1)

    # Castling availability is given one node per side
    for side in "KQkq":
        state_vars.append(1 if side in state.available_castles else 0)

    return state_vars

def get_model():
    inputs = keras.Input(shape=(837,), name="board_state")
    x = layers.Dense(2000, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(2000, activation="relu", name="dense_2")(x)
    x = layers.Dense(2000, activation="relu", name="dense_3")(x)
    x = layers.Dense(2000, activation="relu", name="dense_4")(x)
    outputs = layers.Dense(1, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.BinaryCrossentropy(), #CategoricalCrossentropy
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

if __name__ == "__main__":
    all_games = chess.get_games_data("games/lichess_elite_2019-06.pgn")

    all_board_states = []
    fails = 0

    for game_metadata in tqdm(all_games):
        if game_metadata["Result"] == DRAW:
            continue
        try:
            # Only take that last 5 moves that the white side made
            states = board_states(game_metadata)[::2][-5:]
        except ValueError:
            # In the case that we were not able to parse a move
            # so we assume we messed up somewhere and don't keep any of the states
            # This is ok if it is a small number of games
            states = []
            fails += 1
            continue

        result = game_result(game_metadata)

        for state in states:
            all_board_states.append((state, result))

    print("failed to parse " + str(fails) + "/" + str(len(all_games)) + " games")
    print(len(all_board_states), "States found")
    # Free up some memory
    del all_games

    data = []
    labels = []
    for state, result in tqdm(all_board_states):
        data.append(tf.convert_to_tensor(state_to_int_list(state)))
        labels.append(result)

    del all_board_states

    data_val = tf.convert_to_tensor(data[:10000], dtype=tf.float32)
    data_test = tf.convert_to_tensor(data[10000:20000], dtype=tf.float32)
    data_train = tf.convert_to_tensor(data[20000:], dtype=tf.float32)

    del data

    labels_val = tf.convert_to_tensor(labels[:10000], dtype=tf.float32)
    labels_test = tf.convert_to_tensor(labels[10000:20000], dtype=tf.float32)
    labels_train = tf.convert_to_tensor(labels[20000:], dtype=tf.float32)
    model = get_model()

    print(data_val)

    history = model.fit(
        data_train,
        labels_train,
        batch_size=64,
        epochs=2,
        validation_data=(data_val, labels_val),
    )

    print(history.history)

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(data_test, labels_test, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(data_test[:100])
    print("predictions shape:", predictions.shape)

