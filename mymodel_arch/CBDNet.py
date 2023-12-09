from keras.layers import Conv2D, Add, concatenate, Conv2DTranspose, Input, AveragePooling2D
from keras.models import Model
from utils import psnr, ssim


def autoencoder(optimizer, input_shape=(128, 128, 3)):
    # Input layer
    input_img = Input(input_shape, name="image_input")

    #Noise estimation subnetwork
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(input_img)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(3, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)

    #Non Blind denoising subnetwork
    x = concatenate([x,input_img])
    conv1 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    conv2 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv1)

    pool1 = AveragePooling2D(pool_size=(2,2),padding='same')(conv2)
    conv3 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(pool1)
    conv4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv3)
    conv5 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv4)

    pool2 = AveragePooling2D(pool_size=(2,2),padding='same')(conv5)
    conv6 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(pool2)
    conv7 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv6)
    conv8 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv7)
    conv9 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv8)
    conv10 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv9)
    conv11 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv10)

    upsample1 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(conv11)
    add1 = Add()([upsample1,conv5])
    conv12 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(add1)
    conv13 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv12)
    conv14 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv13)

    upsample2 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(conv14)
    add1 = Add()([upsample2,conv2])
    conv15 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(add1)
    conv16 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv15)

    out = Conv2D(3, (1,1), kernel_initializer='he_normal',padding="same")(conv16)
    out = Add()([out,input_img])

    # Creating the model
    autoencoder = Model(inputs=input_img, outputs=out)
    # Compile the model
    autoencoder.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[ssim, psnr],
    )

    return autoencoder
