import unittest
import chess

class TestMovement(unittest.TestCase):

    def test_castle_kingside(self):
        s = chess.State(FEN_state="4k2r/8/8/8/8/8/8/4K2R w KQkq - 0 1")

        s.move("O-O")
        s.move("O-O")
        self.assertIn("          r k  ", str(s))
        self.assertIn("          R K  ", str(s))


    def test_castle_queenside(self):
        s = chess.State(FEN_state="r3k3/8/8/8/8/8/8/R3K3 w KQkq - 0 1")

        s.move("O-O-O")
        s.move("O-O-O")
        self.assertIn("    k r        ", str(s))
        self.assertIn("    K R        ", str(s))

    def test_promotion(self):
        s = chess.State(FEN_state="7k/P7/8/8/8/8/p7/7K w KQkq - 0 1")
        s.move("a8=Q")
        s.move("a1=Q")
        self.assertIn("Q             k", str(s))
        self.assertIn("q             K", str(s))


if __name__ == '__main__':
    unittest.main()