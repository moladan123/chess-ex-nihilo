from tqdm import tqdm

BISHOP_MOVES = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
ROOK_MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]
KNIGHT_MOVES = [(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (-2, 1), (2, -1), (-2, -1)]

# Used to denote that the piece can move any distance
ARBITRARY_DIST = 8
SINGLE_TILE_MOVE = 1

class Rook:
    speed = ARBITRARY_DIST
    moves = ROOK_MOVES

class Knight:
    speed = SINGLE_TILE_MOVE
    moves = KNIGHT_MOVES

class Bishop:
    speed = ARBITRARY_DIST
    moves = BISHOP_MOVES

class Queen:
    speed = ARBITRARY_DIST
    moves = ROOK_MOVES + BISHOP_MOVES

class King:
    speed = SINGLE_TILE_MOVE
    moves = ROOK_MOVES + BISHOP_MOVES

EMPTY_SPACE = " "
PAWN = 'p'
ROOK = 'r'
KNIGHT = 'n'
BISHOP = 'b'
QUEEN = 'q'
KING = 'k'

ALL_PIECES = "prnbqk"

CASTLE_KINGSIDE = "O-O"
CASTLE_QUEENSIDE = "O-O-O"

# Maps the name to the class containing the basic movement information of a piece
# Excludes pawn movement and castling
NAME_TO_PIECE = {
    ROOK: Rook,
    KNIGHT: Knight,
    BISHOP: Bishop,
    QUEEN: Queen,
    KING: King
}

class State:

    @staticmethod
    def _parse_row(row: str):
        """ Parses one row of the board in FEN format

        :param row: one row of a board in FEN format
            ie. "1pn4Q"
        :return: an array of characters representing the row
            ie. [" ", "p", "n", " ", " ", " ", " ", "Q"]
        """
        final_row = []
        for char in row:

            # any number N expands into N spaces
            if char in "12345678":
                for i in range(int(char)):
                    final_row.append(EMPTY_SPACE)
            else:
                final_row.append(char)

        return final_row

    def __init__(self, FEN_state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        """ Creates a new State,

        :param FEN_state: FEN format string that is used to describe a board state
        defaults to a normal chess start
        """
        board, turn, castle_avail, en_passants, half_moves, full_moves = FEN_state.split()

        # contains all of the pieces
        self.board = list(map(State._parse_row, board.split('/')))

        # Whether the current board state is white to move or not (black to move)
        self.white_to_move = turn == 'w'

        # Which sides can be castled on
        self.available_castles = castle_avail

        # States where it is possible to move en passant, if any
        self.en_passants = '' if en_passants == '-' else en_passants

        # Number of half moves already made in the game.
        # Used to track 50 move repetition
        self.half_moves = int(half_moves)

        # Used the current full move, used to track for algebraic notation
        self.full_moves = int(full_moves)


    def _export_board(self):
        """
        Converts the internal board representation to a FEN standard board position
        Converts all empty spaces to numbers, as per FEN specification
        """
        final_board = []

        for row in self.board:
            final_row = []

            # Tracks the number of continguous empty spaces in the current place
            spaces = 0

            for square in row:
                # we hit an empty space
                if square == EMPTY_SPACE:
                    spaces += 1
                else:
                    final_row.append(str(spaces) + square)
                    spaces = 0

            final_row.append(str(spaces))
            final_row = "".join(final_row).replace('0', '')

            final_board.append(final_row)

        return "/".join(final_board)

    def export_to_FEN(self):
        """
        Exports the current board to an FEN format meant for exporting
        """
        final_FEN_array = []

        # Output state of pieces
        final_FEN_array.append(self._export_board())
        final_FEN_array.append('w' if self.white_to_move else 'b')
        final_FEN_array.append(self.available_castles if self.available_castles else "-")
        final_FEN_array.append(self.en_passants if self.en_passants else "-")
        final_FEN_array.append(str(self.half_moves))
        final_FEN_array.append(str(self.full_moves))

        return " ".join(final_FEN_array)

    @staticmethod
    def _EAN_coords_to_board_coords(EAN_move: str) -> (int, int):
        """
        converts a EAN move to internal coordinates in the board

        :param EAN_move: a 2 letter EAN move to be converted to coordinate form
            eg. "a1" becomes (7, 0)
        :return: a tuple with coordinates to index into the board
            eg if T is the tuple returned,
            the square is accessed by getting  ` board[T[0]][T[1]] `
        """
        assert EAN_move[0] in "abcdefgh" and EAN_move[1] in "12345678", "failed to get " + EAN_move


        col = ord(EAN_move[0]) - ord('a')
        row = 8 - int(EAN_move[1])
        return row, col

    def _get_castling_row(self, piece=''):
        if piece:
            return 7 if piece.isupper() else 0
        return 7 if self.white_to_move else 0

    def parse_EAN(self, EAN: str):
        """
        Converts an extended algebraic notation move to coordinates
            If the move contains a pawn promotion, adds it to extra_info
        :param EAN:
        :return:
        """

        if EAN == CASTLE_KINGSIDE:
            row = self._get_castling_row()
            return (row, 4), (row, 6), CASTLE_KINGSIDE
        elif EAN == CASTLE_QUEENSIDE:
            row = self._get_castling_row()
            return (row, 4), (row, 2), CASTLE_QUEENSIDE

        assert 4 <= len(EAN) <= 5, "Invalid EAN"

        start = State._EAN_coords_to_board_coords(EAN[0:2])
        dest = State._EAN_coords_to_board_coords(EAN[2:4])

        # used to decide what piece to promote to when pawn reaches the end
        extra_info = "" if len(EAN) == 4 else EAN[4]

        return start, dest, extra_info

    @staticmethod
    def _coord_to_EAN(coords):
        """
        Converts a pair of coordinates to
        :param coords:
        :return:
        """
        row, col = coords
        col = chr(col + ord('a'))
        row = str(8 - row)
        return col + row
        return col + row

    @staticmethod
    def convert_to_EAN(start, dest, extra_info=''):
        if extra_info == CASTLE_QUEENSIDE or extra_info == CASTLE_KINGSIDE:
            return extra_info
        return State._coord_to_EAN(start) + State._coord_to_EAN(dest) + extra_info

    @staticmethod
    def _is_same_color(p1: str, p2: str):
        """ Util function that returns true iff the pieces are the same color """
        return p1.islower() == p2.islower()

    def _get_standard_moves_for_piece(self, location, piece):
        valid_moves = set()
        piece_classtype = NAME_TO_PIECE[piece.lower()]
        piece_speed = piece_classtype.speed

        for direction in piece_classtype.moves:
            curr_location = list(location)
            for i in range(piece_speed):
                curr_location[0] += direction[0]
                curr_location[1] += direction[1]

                # if not in range break
                if not (0 <= curr_location[0] <= 7 and 0 <= curr_location[
                    1] <= 7):
                    break

                dest_space = self.board[curr_location[0]][curr_location[1]]

                # Check if the space is occupied by another piece
                # We can only move to that spot if it is occupied by a different color
                if dest_space.lower() in ALL_PIECES:
                    if not State._is_same_color(piece, dest_space):
                        valid_moves.add(State.convert_to_EAN(location, curr_location))
                    break

                # Otherwise we have an empty space and we can keep going
                valid_moves.add(State.convert_to_EAN(location, curr_location))

        return valid_moves

    def _get_pawn_moves(self, location, piece):

        valid_moves = set()

        # moves down if black, moves up if white
        direction = 1 if piece.islower() else -1
        opening_square = 1 if piece.islower() else 6
        promotion_square = 6 if piece.islower() else 1

        # regular movement
        # Move forward by one square
        if self.board[location[0] + direction][location[1]] == EMPTY_SPACE:
            valid_moves.add(State.convert_to_EAN(location, (location[0] + direction, location[1])))

            if location[0] == opening_square and self.board[location[0] + 2 * direction][location[1]] == EMPTY_SPACE:
                valid_moves.add(State.convert_to_EAN(location, (location[0] + 2 * direction, location[1])))

        # Check pawn capture to the left and right side
        for side in (1, -1):
            # Check within bounds and there is a piece there of the opposite color
            if 0 <= location[1] + side <= 7:
                dest = (location[0] + direction, location[1] + side)
                attacked = self.board[dest[0]][dest[1]]

                # Regular capture
                if attacked != EMPTY_SPACE and not State._is_same_color(attacked, piece):
                    valid_moves.add(State.convert_to_EAN(location, dest))

                # En passant capture
                elif State._coord_to_EAN(dest) == self.en_passants:
                    valid_moves.add(State.convert_to_EAN(location, dest))

        # Add all possible pawn promotions if needed
        if location[0] == promotion_square:
            possible_promotions = "rbnq" if piece.islower() else "RBNQ"
            new_valid_moves = set()
            for move in valid_moves:
                for promotion in possible_promotions:
                    new_valid_moves.add(move + promotion)
            valid_moves = new_valid_moves

        return valid_moves

    def _get_possible_castles(self, piece):
        """ Get all possible moves that involve castling

        Currently does not check whether the intermediate square is under check
        #TODO fix that
        """

        valid_moves = set()

        # Cant castle while under check
        if self._check_for_check():
            return valid_moves

        row = self._get_castling_row(piece)
        king = 'K' if self.white_to_move else 'k'
        queen = 'Q' if self.white_to_move else 'q'

        # If there is nothing in the way, we can castle kingside
        if king in self.available_castles: # Does not check if f file is in check
            if self.board[row][5] == self.board[row][6] == EMPTY_SPACE:
                valid_moves.add(CASTLE_KINGSIDE)

        if queen in self.available_castles: # Does not check if d file is in check
            if self.board[row][1] == self.board[row][2] == self.board[row][3] == EMPTY_SPACE:
                valid_moves.add(CASTLE_QUEENSIDE)

        return valid_moves

    def get_all_moves(self, castling_allowed=True):
        """
        Returns a set of EAN moves. This will contain all possible
        valid moves that can be made this turn
        :return:
        """

        can_move = str.isupper if self.white_to_move else str.islower

        valid_moves = set()

        for row_num, row in enumerate(self.board):
            for col_num, piece in enumerate(row):
                if piece != EMPTY_SPACE and can_move(piece):

                    location = (row_num, col_num)

                    # Everything except the pawn movement
                    if piece.lower() in NAME_TO_PIECE:
                        valid_moves = valid_moves.union(self._get_standard_moves_for_piece(location, piece))

                    # Pawn moves
                    if piece.lower() == PAWN:
                        valid_moves = valid_moves.union(self._get_pawn_moves(location, piece))

                    # Castling
                    if castling_allowed and piece.lower() == KING:
                        valid_moves = valid_moves.union(self._get_possible_castles(piece))

        return valid_moves

    def _check_for_check(self):
        """ Return true iff the current king is under attack """

        # Since castling needs to check whether the current king is in check
        # Leaving this to be true would cause infinite recursion
        possible_moves = self.get_all_moves(castling_allowed=False)

        # Find location of opposing king
        king_piece = 'k' if self.white_to_move else 'K'
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == king_piece:
                    location = (i, j)

        # Check if there is a piece that can capture the king,
        # By checking if there is something that can move into a king's square
        king_location = State._coord_to_EAN(location)
        for move in possible_moves:
            if king_location in move:
                return True

        return False

    def _invalidate_castles(self):
        """ Invalidates castling if either the king or rook was moved """

        if self.board[0][0] != 'r': # Black queenside
            self.available_castles = self.available_castles.replace('q', '')

        if self.board[0][7] != 'r': # Black kingside
            self.available_castles = self.available_castles.replace('k', '')

        if self.board[7][0] != 'R': # White queenside
            self.available_castles = self.available_castles.replace('Q', '')

        if self.board[7][7] != 'R': # White kingside
            self.available_castles = self.available_castles.replace('K', '')

        if self.board[0][4] != 'k':
            self.available_castles = self.available_castles.replace('k', '')
            self.available_castles = self.available_castles.replace('q', '')

        if self.board[7][4] != 'K':
            self.available_castles = self.available_castles.replace('K', '')
            self.available_castles = self.available_castles.replace('Q', '')

    def _update_en_passant(self, start, dest, piece):
        """ Updates the board if the move is en passant """
        if piece.lower() != 'p':
            self.en_passants = ''
            return

        # check if the pawn moves by two spaces
        if abs(start[0] - dest[0]) == 2:
            en_passant_square = (start[0] + dest[0]) // 2
            self.en_passants = State._coord_to_EAN((en_passant_square, start[1]))
            return

        # Check if piece is eating via en passant, make sure to explicitly make the capture
        if State._coord_to_EAN(dest) == self.en_passants:
            self.board[start[0]][dest[1]] = EMPTY_SPACE

    def _update_board(self, start: (int, int), dest: (int, int), extra_info=''):
        """ updates self.board to with the given move """

        piece = self.board[start[0]][start[1]]

        # Move the piece itself
        self.board[dest[0]][dest[1]] = piece
        self.board[start[0]][start[1]] = EMPTY_SPACE

        # Special moves
        if extra_info:

            # Pawn promotion is always in the form /=[QBNRqbnr]/
            if extra_info[0] == "=":
                self.board[dest[0]][dest[1]] = extra_info[1]

            # Castling kingside
            elif extra_info == CASTLE_KINGSIDE:
                row = self._get_castling_row()

                # We already moved the king, so we just need to move the rook
                self.board[row][5] = self.board[row][7]
                self.board[row][7] = EMPTY_SPACE

            elif extra_info == CASTLE_QUEENSIDE:
                row = self._get_castling_row()

                # King already moved, so just update the rook
                self.board[row][3] = self.board[row][0]
                self.board[row][0] = EMPTY_SPACE

        # en passant
        self._update_en_passant(start, dest, piece)

    def _move(self, start: (int, int), dest: (int, int), extra_info=''):
        """ Attempts to move the piece from the state to the dest coordinates

        :param start: The coordinates of the piece that is being moved
        :param dest: The coordinates of the destination move
        :param extra_info:
        :return:
        """
        moving_piece = self.board[start[0]][start[1]]
        end_piece = self.board[dest[0]][dest[1]]

        # Check if the move is valid
        possible_moves = self.get_all_moves()
        if State.convert_to_EAN(start, dest, extra_info) not in possible_moves:
            return False

        # Invalidate castling
        self._invalidate_castles()

        # Update half turn counters since capture (updates 50 move draw)
        # reset on capture, which is when the destination piece is a different color
        self.half_moves += 1
        if not State._is_same_color(moving_piece, end_piece):
            self.half_moves = 0

        # Update full moves after black's turn
        if not self.white_to_move:
            self.full_moves += 1

        # Update the board to reflect the move
        self._update_board(start, dest, extra_info)

        # Update move history TODO
        # Detect three move repetition TODO

        # Update whose turn it is
        self.white_to_move = not self.white_to_move

    def move(self, AN_str):
        """ Makes the move encoded in standard algebraic notation """
        self._move(*self._AN_to_coords(AN_str))

    def _AN_to_coords(self, move: str):
        """ Converts an algebraic notation move to internal coordinates """

        orig_move = move

        extra_info = ""

        # remove all characters that don't matter when parsing
        for pointless_char in "x+#":
            move = move.replace(pointless_char, "")

        # Handle castling
        if CASTLE_QUEENSIDE in move:
            row = self._get_castling_row()
            return (row, 4), (row, 2), CASTLE_QUEENSIDE
        elif CASTLE_KINGSIDE in move:
            row = self._get_castling_row()
            return (row, 4), (row, 6), CASTLE_KINGSIDE

        # Pawn promotion
        if move[-2] == "=":
            extra_info = move[-1] if self.white_to_move else move[-1].lower()
            move = move[:-2]

        # Destination of move, this is the only guaranteed substring in the move
        dest_str = move[-2:]
        dest = State._EAN_coords_to_board_coords(dest_str)
        move = move[:-2]

        # Deduce what piece actually made the move, if there is no shown there is no pawn
        # Note in AN pieces are always uppercase and location is lowercase,
        # so this makes it simple to check if we have a piece or a location
        piece = "P"
        if move and move[0].isupper():
            piece = move[0]
            move = move[1:]
        if not self.white_to_move:
            piece = piece.lower()

        # At this point the only info the move should contain is a hint on where the piece is coming from
        loc_hint = move

        possible_moves = self.get_all_moves()
        possible_moves = filter(lambda x: dest_str in x, possible_moves) # Filter to only moves that land on the right destination
        possible_moves = list(filter(lambda x: loc_hint in x[0:2], possible_moves)) # Filter to only moves that match the hint in the algebraic notation
        for possible_move in possible_moves:
            row, col = State._EAN_coords_to_board_coords(possible_move[0:2])
            if self.board[row][col] == piece:
                return (row, col), dest, extra_info

        raise ValueError("Algebraic notation parsing failed, no valid move found matching the given move " + orig_move
                         + " with board state\n" + str(self))


    def __str__(self):
        ret = "White to move\n" if self.white_to_move else "Black to move\n"
        for index, row in enumerate(self.board):
            ret += str(8 - index) + "\t" + " ".join(row) + "\n"
        ret += "\n \tA B C D E F G H"
        return ret

def get_games_data(pgn_filename):
    with open(pgn_filename) as pgn_file:
        all_lines = pgn_file.readlines()

    # For each game, make a dictionary with all the information
    all_games = []
    curr_game_metadata = dict()
    curr_game_moves = []

    metadata = True
    for line in tqdm(all_lines):

        # Metadata on the game (can probably be thrown away if we don't care about it)
        if metadata:
            if line == '\n':
                metadata = False
                curr_game_moves = []
            else:
                words = line.strip().split(" ", 1)
                curr_game_metadata[words[0][1:]] = words[1][1:-2]

        else:  # The actual moves of the game
            if line == '\n':
                metadata = True

                # get all the moves and exclude the ones that are just number labels
                curr_game_moves = "".join(curr_game_moves)
                moves = list(filter(lambda x: "." not in x, curr_game_moves.split(" ")))
                curr_game_metadata['result'] = moves[-1]
                curr_game_metadata['moves'] = moves[:-1]

                # Reset all values
                all_games.append(curr_game_metadata)
                curr_game_metadata = dict()
                curr_game_moves = []

            else:
                # Keep reading input until we are done
                curr_game_moves += line.rstrip() + " "

    return all_games




if __name__ == "__main__":
    s = State()

    # Done using games downloaded from https://database.lichess.org/
    # The files are far too large to put in a git repository

    all_games = get_games_data("games/lichess_elite_2020-05.pgn")
    print(len(all_games), "total games detected")
