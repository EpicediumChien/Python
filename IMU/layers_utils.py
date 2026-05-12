import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv1D, Dropout, LayerNormalization, MultiHeadAttention, Add)

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        # 在初始化時直接建立權重，這對 Keras 的 load_model 最友好
        self.pos_weight = self.add_weight(
            name="pos_weight",
            shape=(sequence_length, embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (Batch, Seq_Len, Embed_Dim)
        return inputs + self.pos_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])
    
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return Add()([x, res])