from train_util import BatchGenerator
from tokenizer import Tokenizer
from torch import Tensor
from torch import int32

class TrainUtilTests(unittest.TestCase):
    
    # Setting up the tests.
    path = "test_data.txt"
    with open(path, "r") as f:
        text = f.read()
    tk = Tokenizer()
    data = tk.encode(text)
    block_size = 8
    batch_size = 4
    bg = BatchGenerator(data, block_size = block_size, batch_size = batch_size)

    def test_correct_type(self):
        batch = bg.get_batch()
        self.assertEqual(type(batch), Tensor)

    def test_correct_dims(self):
        batch = bg.get_batch()
        self.assertEqual(batch.size(dim=0), 4)
        self.assertEqual(batch.size(dim=1), 8)

    def test_correct_dtype(self):
        batch = bg.get_batch()
        self.assertEqual(batch.dtype, int32)
