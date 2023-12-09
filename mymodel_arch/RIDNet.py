import tensorflow as tf
from keras.layers import (
    Conv2D,
    Input,
    Conv2DTranspose,
    Add,
    GlobalAveragePooling2D,
    concatenate,
    Activation,
    Reshape,
    Multiply,
)
from keras.models import Model
from utils import psnr, ssim


class EAM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = Conv2D(
            64, (3, 3), dilation_rate=1, padding="same", activation="relu"
        )
        self.conv2 = Conv2D(
            64, (3, 3), dilation_rate=2, padding="same", activation="relu"
        )

        self.conv3 = Conv2D(
            64, (3, 3), dilation_rate=3, padding="same", activation="relu"
        )
        self.conv4 = Conv2D(
            64, (3, 3), dilation_rate=4, padding="same", activation="relu"
        )

        self.conv5 = Conv2D(64, (3, 3), padding="same", activation="relu")

        self.conv6 = Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv7 = Conv2D(64, (3, 3), padding="same")

        self.conv8 = Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv9 = Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv10 = Conv2D(64, (1, 1), padding="same")

        self.gap = GlobalAveragePooling2D()

        self.conv11 = Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv12 = Conv2D(64, (3, 3), padding="same", activation="sigmoid")

    def call(self, input):
        conv1 = self.conv1(input)
        conv1 = self.conv2(conv1)

        conv2 = self.conv3(input)
        conv2 = self.conv4(conv2)

        concat = concatenate([conv1, conv2])
        conv3 = self.conv5(concat)
        add1 = Add()([input, conv3])

        conv4 = self.conv6(add1)
        conv4 = self.conv7(conv4)
        add2 = Add()([conv4, add1])
        add2 = Activation("relu")(add2)

        conv5 = self.conv8(add2)
        conv5 = self.conv9(conv5)
        conv5 = self.conv10(conv5)
        add3 = Add()([add2, conv5])
        add3 = Activation("relu")(add3)

        gap = self.gap(add3)
        gap = Reshape((1, 1, 64))(gap)
        conv6 = self.conv11(gap)
        conv6 = self.conv12(conv6)

        mul = Multiply()([conv6, add3])
        out = Add()([input, mul])  # This is not included in the reference code
        return out


def autoencoder(optimizer, input_shape=(128, 128, 3)):
    # Input layer
    input_img = Input(input_shape, name="image_input")

    conv1 = Conv2D(64, (3, 3), padding="same")(input_img)
    eam1 = EAM()(conv1)
    eam2 = EAM()(eam1)
    eam3 = EAM()(eam2)
    conv2 = Conv2D(3, (3, 3), padding="same")(eam3)
    out = Add()([conv2, input_img])
    autoencoder = Model(inputs=input_img, outputs=out)
    # Compile the model
    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[ssim, psnr],
    )

    return autoencoder
