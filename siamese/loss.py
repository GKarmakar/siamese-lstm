from keras.layers.merge import Add
import keras.backend as K


def energy():
    pass


def energy_loss(y_true, y_pred):
    pass


class Diff(Add):
    def _merge_function(self, inputs):
        return K.sum(K.abs(inputs[1] - inputs[0]))

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        return None, 1

    def compute_output_shape(self, input_shape):
        return None, 1


class CosineDist(Add):
    def _merge_function(self, inputs):
        l, r = inputs[0], inputs[1]
        num = K.sum((l * r), keepdims=True, axis=-1)
        den = K.sqrt(K.sum(K.square(l), keepdims=True, axis=-1)) * K.sqrt(K.sum(K.square(r), keepdims=True, axis=-1))
        den = K.clip(den, min_value=1e-4, max_value=float('inf'))
        sim = num / den
        return K.ones_like(sim) - sim

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        return None, 1

    def compute_output_shape(self, input_shape):
        return None, 1
