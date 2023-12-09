from keras.models import Model
from keras.layers import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Input,
    concatenate,
)

from utils import psnr, ssim


def autoencoder(optimizer):
    input_img = Input(shape=(128, 128, 3))

    # Encoding layers
    conv1 = Conv2D(128, (3, 3), activation="relu", padding="same")(input_img)
    conv1 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2), padding="same")(conv1)

    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2), padding="same")(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2), padding="same")(conv3)

    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv4)

    # Decoding layers
    up1 = UpSampling2D((2, 2))(conv4)
    concat1 = concatenate([conv3, up1])
    conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat1)
    conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv5)

    up2 = UpSampling2D((2, 2))(conv5)
    concat2 = concatenate([conv2, up2])
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat2)
    conv6 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv6)

    up3 = UpSampling2D((2, 2))(conv6)
    concat3 = concatenate([conv1, up3])
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    # Output layer
    output_layer = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(conv7)

    # Model
    u_net = Model(inputs=input_img, outputs=output_layer)
    u_net.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[psnr, ssim],  # Add appropriate metrics
    )

    return u_net
