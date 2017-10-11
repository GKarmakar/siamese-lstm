from keras.callbacks import Callback


class LSTMCallback(Callback):
    def __init__(self, lstm, metric='loss'):
        Callback.__init__(self)
        self.lstm = lstm
        self.metric = metric
        self.best_metric = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs[self.metric] < self.best_metric:
            self.best_metric = logs[self.metric]
            print("\nSaving model to %s..." % self.lstm.directory, end=' ')
            self.lstm.save()
            print('Done.\n')
        else:
            print('\nNo improvement on metric. Skipping save...\n')
