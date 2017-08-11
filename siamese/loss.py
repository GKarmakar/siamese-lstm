from keras.layers.merge import Add
import keras.backend as K


def energy():
    pass


def energy_loss(y_true, y_pred):
    pass


class Diff(Add):
    def _merge_function(self, inputs):
        return K.abs(inputs[1] - inputs[0])
