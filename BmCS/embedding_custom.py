import keras.backend as K
from keras.layers import Embedding

class EmbeddingWithDropout(Embedding):

    def __init__(self, dropout_rate, *args, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        _embeddings = K.in_train_phase(K.dropout(self.embeddings, self.dropout_rate, noise_shape=[self.input_dim,1]), self.embeddings) if self.dropout_rate > 0 else self.embeddings
        out = K.gather(_embeddings, inputs)
        return out

    def get_config(self):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
