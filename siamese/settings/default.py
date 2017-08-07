# Data settings
SENTENCE_LEN = 100
PATHS = ('data/amazon/train.tsv', 'data/amazon/test.tsv')
EMBED_TYPE = 'hot_coded'
EMBED_DIM = None
LINE_LIMIT = (10, 10)
POS_VALUE = 0.0
NEG_VALUE = 1.0

# Model settings
RECURRENT_NEURONS = (512, 512, 128)
DENSE_UNITS = 128
DROPOUT = 0.3

# Training settings
EPOCHS = 50
BATCH_SIZE = 30
LEARNING_RATE = 0.001
