# Data settings
SENTENCE_LEN = 50
FASTTEXT_PATH = 'data/fasttext/vi'
PATHS = ('data/intent_corpus2/train.tsv', 'data/intent_corpus2/test.tsv', 'data/intent_corpus2/test.tsv')
LINE_LIMIT = (10, 10, None)
POS_VALUE = 0.0
NEG_VALUE = 1.0
EMBED_DIMS = 100

# Model settings
RECURRENT_NEURONS = (128, 128, 128)
DENSE_UNITS = 128
DROPOUT = 0.0
RECURRENT_REGULARIZER = 0.0
DENSE_REGULARIZER = 0.0
MERGE_LAYER = 'diff'

# Training settings
EPOCHS = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.001
