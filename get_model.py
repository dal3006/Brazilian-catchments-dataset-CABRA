from os import name
import tensorflow as tf
from tensorflow import keras
from keras.layers import GlobalAveragePooling1D,Bidirectional, LSTM, Dropout, Dense,Input,LayerNormalization,MultiHeadAttention,Conv1D
from keras.models import Model

def lstm_model(input_shape,lstm_units,lstm_dropout,unit):
    input1 = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units), name='bidirectional_lstm1')(input1)
    x = Dropout(rate=lstm_dropout)(x)
    x = Dense(units=unit, kernel_initializer=tf.initializers.zeros, name='dense1')(x)

    model = Model(input1,x)

    return model

def lstm_att(input_shape,lstm_units,lstm_dropout,unit):
    input1 = Input(shape=input_shape)
    x = Bidirectional(LSTM(lstm_units,return_sequences=True,activation="relu"), name='bidirectional_lstm1')(input1)
    x = Dropout(rate=lstm_dropout)(x)
    x,att_scr = Attention()([x,x],return_attention_scores=True)
    #x,att_scr = MultiHeadAttention(key_dim=128, num_heads=6, dropout=0.2)(x, x,return_attention_scores=True)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Dropout(rate=lstm_dropout)(x)

    x = Dense(units=unit, kernel_initializer=tf.initializers.zeros, name='dense1')(x)

    model = Model(input1,x)

    return model








def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res



def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="relu")(x)
    return Model(inputs, outputs)


