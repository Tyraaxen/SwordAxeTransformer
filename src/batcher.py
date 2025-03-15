import os
import jax.numpy as jnp
import jax.random as jr

class Batcher:
    
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        
        self.block_size = 8
        self.batch_size = 4
        
    def get_block_size(self):
        return self.block_size
    
    def get_batch_size(self):
        return self.batch_size
        
    def get_batch(self, key, split):
        # Choose data set and validate input
        if split == "train":
            data = self.train_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f'Value of split must be either "train" or "test". Got {split}.')
        return self._get_batch(key, data, self.block_size, self.batch_size)

    def _get_batch(self, key, data, block_size, batch_size):
    
        # TODO: Currently samples with replacement. Fix to sampling without replacement. Use choice.
        # Generates indices in the range [0, len(data))
        idxs = jr.randint(key, shape = (batch_size,), minval = 0, maxval = len(data) - block_size) 
        # TODO: Consider using vmap here. Check with GPT.
        x = jnp.stack(jnp.array([data[i:i+self.block_size] for i in idxs]))
        # +1 will not throw index error since idxs does not include the maxval but only up until maxval-1
        y = jnp.stack(jnp.array([data[i+1:i+self.block_size+1] for i in idxs]))
        return x, y
    
if __name__ == "__main__":
    from tokenizer import Tokenizer
    path = "test_data.txt"
    with open(os.getcwd() + "\\tests\\" + path, "r", encoding = "utf-8") as f:
        text = f.read()
    tk = Tokenizer()
    data = tk.encode(text)
    
    batcher = Batcher(data, [])
    key = jr.PRNGKey(42)
    x, y = batcher.get_batch(key, "train")
    print(x)
    print(y)