from keras.layers import Input, Conv2D, Add, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from utils import psnr, ssim


def autoencoder(optimizer):
    input_img = Input(shape=(128, 128, 3), name="image_input")

    # Residual Blocks
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    # Adding the residual connection
    residual = Add()([x, input_img])

    # Convolutional layer to adjust the shape
    residual = Conv2D(64, (3, 3), activation="relu", padding="same")(residual)
    residual = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(residual)

    # Output
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(residual)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[ssim, psnr],
    )

    return autoencoder
