import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    GaussianNoise,
    Input,
    LSTM,
    MaxPooling1D,
    concatenate,
)


def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.02)(inputs)

    x = Conv1D(64, 7, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)

    avg_p = GlobalAveragePooling1D()(x)
    max_p = GlobalMaxPooling1D()(x)
    combined = concatenate([avg_p, max_p])

    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="LSTM_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model
