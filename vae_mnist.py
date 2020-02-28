from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


def sampling(args: tuple):
    _z_mean, _z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(_z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return _z_mean + K.exp(_z_log_var / 2) * epsilon


x = Input(shape=(original_dim,), name="input")
h = Dense(intermediate_dim, activation="relu", name="encoding")(x)
# 潜在空間の平均を定義
z_mean = Dense(latent_dim, name="mean")(h)
# 潜在空間でのログ分散を定義
z_log_var = Dense(latent_dim, name="log_variance")(h)
z = Lambda(sampling,output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")

input_decoder = Input(shape=(latent_dim,), name="decoder_input")
decoder_h = Dense(intermediate_dim, activation="relu", name="decoder_h")(input_decoder)
x_decoded = Dense(original_dim, activation="sigmoid", name="flat_decoded")(decoder_h)
decoder = Model(input_decoder, x_decoded, name="decoder")
decoder.summary()

output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()
