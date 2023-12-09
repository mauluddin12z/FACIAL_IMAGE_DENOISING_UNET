from keras.models import Model
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Input,
)

from utils import psnr, ssim


def autoencoder(optimizer):
    input_img = Input(shape=(128, 128, 3))

    # Encoding layers
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(input_img)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same")(x)

    # Decoding layers
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)

    # Output layer
    x = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[
            ssim,
            psnr,
        ],
    )

    return autoencoder
