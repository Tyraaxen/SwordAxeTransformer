# Class used to help with testing if an error was raised or not.
class NotRaise:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise AssertionError(f"An unexpected exception was raised: {exc_value}")
        return True  # Suppress the exception for testing purposes