class Tokenizer:
    def __init__(self):
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:',.<>?/`~ \t\n\r"
        self.stoi = {s:i for i,s in enumerate(self.chars)}
        self.itos = {i:s for i,s in enumerate(self.chars)}
        
    def encode(self, string):
        return [self.stoi[i] for i in string]
    
    def decode(self, tokenz):
        return ''.join([self.itos[i] for i in tokenz])
    