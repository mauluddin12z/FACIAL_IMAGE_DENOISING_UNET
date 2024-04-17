import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


def preprocess_and_load_data(dir, file_names):
    data = []
    for filename in file_names:
        try:
            img = cv2.imread(os.path.join(dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = img.astype("float32") / 255.0
                data.append(img)
            else:
                print(f"Warning: Skipping {filename} due to reading error.")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    return np.array(data)


def custom_range(start, stop, step):
    current = start
    result = []
    while current <= stop + step / 2:
        result.append(round(current, 2))
        current += step
    return result


def add_blur(images, range_set):
    odd_numbers = [
        round(i, 2)
        for i in np.arange(range_set[0], range_set[1], range_set[2])
        if i != 0
    ]
    num_images = len(images)
    total_odd_numbers = len(odd_numbers)
    blurred_images = []

    for i in range(num_images):
        odd_number = odd_numbers[i % total_odd_numbers]
        ksize = (odd_number, odd_number)
        blurred_image = cv2.blur(images[i], ksize, 0)
        blurred_images.append(blurred_image)

    return np.array(blurred_images)


def add_gaussian_blur(images, range_set):
    odd_numbers = [
        round(i, 2)
        for i in np.arange(range_set[0], range_set[1], range_set[2])
        if i != 0
    ]
    num_images = len(images)
    total_odd_numbers = len(odd_numbers)
    blurred_images = []

    for i in range(num_images):
        odd_number = odd_numbers[i % total_odd_numbers]
        ksize = (odd_number, odd_number)
        blurred_image = cv2.GaussianBlur(images[i], ksize, 0)
        blurred_images.append(blurred_image)

    return np.array(blurred_images)


def add_noise(images, range_set):
    numbers = custom_range(range_set[0], range_set[1], range_set[2])
    num_images = len(images)
    num_noise_levels = len(numbers)
    noisy_images = []
    rng = np.random.default_rng()

    for i in range(num_images):
        noise_level = numbers[i % num_noise_levels]
        noise = noise_level * rng.normal(loc=0.0, scale=1.0, size=images[i].shape)
        noisy_image = np.clip(images[i] + noise, 0.0, 1.0)
        noisy_images.append(noisy_image)

    return np.array(noisy_images)


def display_image(*arrays, labels=None, num_samples=10, figsize=(20, 5), cmap=None):
    if labels is None:
        labels = list(range(1, len(arrays) + 1))
    num_sets = len(arrays)
    plt.figure(figsize=figsize)
    for i in range(num_samples):
        for j in range(num_sets):
            if i < len(arrays[j]):
                ax = plt.subplot(num_sets, num_samples, j * num_samples + i + 1)
                plt.imshow(arrays[j][i], cmap=cmap)
                if labels is not None:
                    ax.set_title(labels[j])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

    plt.show()


def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


def checkpoint_callback(cp_dir):
    checkpoint = ModelCheckpoint(
        cp_dir,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    return checkpoint


def early_stopping_callback():
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, mode="min"
    )
    return early_stopping


def tensorboard_callback(base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)

    existing_experiment_nums = [
        int(d.split("-")[-1])
        for d in os.listdir(base_log_dir)
        if d.startswith("experiment")
    ]

    next_experiment_num = (
        max(existing_experiment_nums) + 1 if existing_experiment_nums else 1
    )
    experiment_dir = f"experiment-{next_experiment_num}"

    log_dir = os.path.join(base_log_dir, experiment_dir)
    os.makedirs(log_dir)

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    return log_dir, experiment_dir, tensorboard_callback


def create_experiment_notes(**kwargs):
    experiment_notes = (
        "| Key                 | Value                   |\n"
        "| -------------------- | ----------------------- |\n"
    )

    for key, value in kwargs.items():
        experiment_notes += f"| {key}                 | {value}                   |\n"
    return experiment_notes
