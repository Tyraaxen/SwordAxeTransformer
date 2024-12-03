import unittest
from tokenizer import Tokenizer

# Class used to help with testing if an error was raised or not.
class NotRaise:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise AssertionError(f"An unexpected exception was raised: {exc_value}")
        return True  # Suppress the exception for testing purposes

class TokenizerTests(unittest.TestCase):
    tokenizer = Tokenizer()

    def test_single_char(self):
        self.assertEqual(self.tokenizer.encode("a"), [0])

    def test_multiple_chars(self):
        self.assertEqual(self.tokenizer.encode("tyrone"), [19, 24, 17, 14, 13, 4])

    def test_identity_condition_ints(self):
        # Here we want to see that encode ∘ decode(ints) = ints 
        test_ints = [19,24,17,14,13,4]
        identity = lambda ints: self.tokenizer.encode(self.tokenizer.decode(ints))
        self.assertEqual(identity(test_ints), test_ints)

    def test_identity_condition_string(self):
        # Here we want to see that decode ∘ encode(string) = string 
        test_string = "dampunge"
        identity = lambda string: self.tokenizer.decode(self.tokenizer.encode(string))
        self.assertEqual(identity(test_string), test_string)

    def test_special_characters(self):
        # Make sure we don't fail with nordic + other special chars
        with NotRaise():
            self.assertEqual(self.tokenizer.encode("åäö&%/"), [26, 27, 28, 74, 72, 95])

    def test_encode_throw_if_not_ints(self):
        with self.assertRaises(TypeError):
            # Give wrong type and expect an informative error message
            self.tokenizer.encode([1,2,3])

    def test_decode_throw_if_not_str(self):
        with self.assertRaises(TypeError):
            # Give wrong type and expect an informative error message
            self.tokenizer.decode("Hoppsan!")

if __name__ == '__main__':
    unittest.main()