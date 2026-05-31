import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    GaussianNoise,
    Input,
    MaxPooling1D,
    concatenate,
)
from tensorflow.keras.regularizers import l2

from layers_utils import PositionalEmbedding, transformer_encoder


def build_model(input_shape, seq_len=50, embed_dim=64):
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.05)(inputs)

    x = Conv1D(64, 7, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = PositionalEmbedding(seq_len, embed_dim)(x)
    x = transformer_encoder(x, head_size=embed_dim, num_heads=2, ff_dim=embed_dim, dropout=0.5)

    avg_p = GlobalAveragePooling1D()(x)
    max_p = GlobalMaxPooling1D()(x)
    combined = concatenate([avg_p, max_p])

    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-3))(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="Transformer_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model
