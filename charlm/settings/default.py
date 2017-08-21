# Data settings
SENTENCE_LEN = 150
PATHS = ('data/pararomance/train.txt', 'data/pararomance/test.txt')
EMBED_TYPE = 'hot_coded'
EMBED_DIM = None
LINE_LIMIT = (100, 10)

# Model settings
RECURRENT_NEURONS = (1000, 500,)
DROPOUT = 0.5
CONV_FILTERS = 100
CONV_KERNEL = 3

# Training settings
EPOCHS = 5
BATCH_SIZE = 30
LEARNING_RATE = 0.001
