class Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.epochs = 100
        self.batch_size = 512
        self.num_classes = 10

# Create an instance for easy import
config = Config()