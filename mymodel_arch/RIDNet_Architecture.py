import tensorflow as tf
from keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    Concatenate,
    Add,
    Activation,
    Reshape,
    Multiply,
    Input,
)
from keras.models import Model

from utils import ssim, psnr


def autoencoder(optimizer):
    def eam(input):
        x = Conv2D(64, (3, 3), dilation_rate=1, padding="same", activation="relu")(
            input
        )
        x = Conv2D(64, (3, 3), dilation_rate=2, padding="same", activation="relu")(x)

        y = Conv2D(64, (3, 3), dilation_rate=3, padding="same", activation="relu")(
            input
        )
        y = Conv2D(64, (3, 3), dilation_rate=4, padding="same", activation="relu")(y)

        z = Concatenate(axis=-1)([x, y])
        z = Conv2D(64, (3, 3), padding="same", activation="relu")(z)
        add_1 = Add()([z, input])

        z = Conv2D(64, (3, 3), padding="same", activation="relu")(add_1)
        z = Conv2D(64, (3, 3), padding="same")(z)
        add_2 = Add()([z, add_1])
        add_2 = Activation("relu")(add_2)

        z = Conv2D(64, (3, 3), padding="same", activation="relu")(add_2)
        z = Conv2D(64, (3, 3), padding="same", activation="relu")(z)
        z = Conv2D(64, (1, 1), padding="same")(z)
        add_3 = Add()([z, add_2])
        add_3 = Activation("relu")(add_3)

        z = GlobalAveragePooling2D()(add_3)
        z = Reshape((1, 1, 64))(z)
        z = Conv2D(4, (3, 3), padding="same", activation="relu")(z)
        z = Conv2D(64, (3, 3), padding="same", activation="sigmoid")(z)
        mul = Multiply()([z, add_3])

        return mul

    input_img = Input(shape=(128, 128, 3), name="input")
    feat_extraction = Conv2D(64, (3, 3), padding="same")(input_img)
    eam_1 = eam(feat_extraction)
    eam_2 = eam(eam_1)
    eam_3 = eam(eam_2)
    eam_4 = eam(eam_3)
    x = Conv2D(3, (3, 3), padding="same")(eam_4)
    add_2 = Add()([x, input_img])

    ridnet = Model(input_img, add_2)

    ridnet.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[
            ssim,
            psnr,
        ],
    )
    return ridnet
