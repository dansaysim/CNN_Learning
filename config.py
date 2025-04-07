class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.epochs = 20
        self.batch_size = 128
        self.num_classes = 10

# Create an instance for easy import
config = Config()