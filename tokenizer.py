from typing import List

class Tokenizer:
    def __init__(self):
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:',.<>?/`~ \t\n\r"
        self.stoi = {s:i for i,s in enumerate(self.chars)}
        self.itos = {i:s for i,s in enumerate(self.chars)}
        
    def encode(self, string: str) -> List[int]:
        # To help if wrong input is given so a nice error message is printed
        if not isinstance(string, str):
            raise TypeError(f"Wrong type {type(str)} for decode. Input must be a string.")
        return [self.stoi[i] for i in string]
    
    def decode(self, tokenz: List[int]) -> str:
        # Same here
        if not isinstance(tokenz, list):
            raise TypeError(f"Wrong type {type(tokenz)} for decode. Input must be an iterable.")
        return ''.join([self.itos[i] for i in tokenz])
    