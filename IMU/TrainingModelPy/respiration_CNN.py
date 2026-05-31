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
    SpatialDropout1D,
    concatenate,
)
from tensorflow.keras.regularizers import l2


def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.02)(inputs)

    x1 = Conv1D(64, 15, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x1 = BatchNormalization()(x1)
    x1 = SpatialDropout1D(0.4)(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Conv1D(128, 7, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.3)(x2)

    x3 = Conv1D(128, 3, padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)

    avg_p = GlobalAveragePooling1D()(x3)
    max_p = GlobalMaxPooling1D()(x3)
    combined = concatenate([avg_p, max_p])

    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Respiration_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model
