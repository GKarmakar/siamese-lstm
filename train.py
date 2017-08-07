from siamese.loader import TwinLoader
from siamese.model import LSTMSiameseNet
from siamese.settings.default import *


print('Loading data...')
loader = TwinLoader(paths=PATHS, path_alias=('train', 'test'), nlines=LINE_LIMIT,
                    embed_type=EMBED_TYPE, embed_dims=EMBED_DIM, sentence_len=SENTENCE_LEN)
loader.load_data()

print('Creating model...')
model = LSTMSiameseNet(loader, dense_units=DENSE_UNITS, recurrent_neurons=RECURRENT_NEURONS,
                       dropout=DROPOUT)
print('Compiling...')
model.compile(learning_rate=LEARNING_RATE)
print('Starting training...')
model.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
