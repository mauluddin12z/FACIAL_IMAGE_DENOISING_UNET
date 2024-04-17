from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, MaxPooling2D
from utils import psnr, ssim


def conv_block(x, filters, kernel_size=(3, 3), activation="relu", padding="same"):
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding)(x)
    return x


def upsample_block(
    x, skip_connection, filters, kernel_size=(3, 3), strides=(2, 2), padding="same"
):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = concatenate([skip_connection, x])
    x = conv_block(x, filters)
    return x


def build_encoder(x, filters, levels=5):
    skips = []

    for _ in range(levels):
        x = conv_block(x, filters)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

    return x, skips


def build_decoder(x, skips, filters, levels=5):
    for i in reversed(range(levels)):
        x = upsample_block(x, skips[i], filters)

    return x


def autoencoder(optimizer, input_shape=(128, 128, 3)):
    filters = 128
    input_img = Input(input_shape, name="image_input")

    x, skips = build_encoder(input_img, filters)
    x = conv_block(x, filters)
    x = build_decoder(x, skips, filters)

    # Output Layer
    output_layer = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)

    autoencoder = Model(inputs=input_img, outputs=output_layer)
    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[ssim, psnr],
    )

    return autoencoder
