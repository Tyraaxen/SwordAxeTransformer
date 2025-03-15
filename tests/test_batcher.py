import unittest
import sys
import os
import jax.numpy as jnp
import jax.random as jr

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, parent_dir)

from batcher import Batcher
from tokenizer import Tokenizer
from test_util import NotRaise

class BatcherTests(unittest.TestCase):
        # Setting up the tests.
    path = os.getcwd() + "\\tests\\" + "test_data.txt"
    with open(path, "r", encoding = "utf-8") as f:
        text = f.read()
    tk = Tokenizer()
    data = tk.encode(text)
    batcher = Batcher(data, [])
    
    def test_throws_on_invalid_split(self):  
        key = jr.PRNGKey(42)
        bad_input = "validation"
        with self.assertRaises(ValueError):
            self.batcher.get_batch(key, bad_input)
            
    def test_size_is_consistent(self):
        key = jr.PRNGKey(1337)
        with NotRaise():
            x, y = self.batcher.get_batch(key, "train") 
            # Check x has right shape
            self.assertEqual(x.shape, (self.batcher.get_batch_size(), self.batcher.get_block_size()))
            # Check y has right shape
            self.assertEqual(y.shape, (self.batcher.get_batch_size(), self.batcher.get_block_size()))           
            
    def test_data_and_label_is_correct(self):

        key = jr.PRNGKey(1337) # The seed really doesn't matter here but we need a key
        x, y = self.batcher.get_batch(key, "train")
        
        # Check that the data has the correct structure
        for i in range(self.batcher.get_batch_size()):
            self.assertEqual(y[i][0], x[i][1])
        