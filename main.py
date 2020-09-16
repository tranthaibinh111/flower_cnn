# region Python
import os
import pathlib
# endregion

# region TensorFlow
import tensorflow as tf
# import keras (high level API) with tensorflow as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# endregion

# region Matplotlib
import matplotlib.pyplot as plt
# endregion
# endregion


def load_data():
    r"""
    https://www.tensorflow.org/tutorials/load_data/images#download_the_flowers_dataset
    :return:
    """
    # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    # data_dir = tf.keras.utils.get_file(origin=dataset_url,
    #                                    fname='flower_photos',
    #                                    untar=True)
    data_dir = pathlib.Path(r"{0}\flower_photos".format(os.getcwd()))

    return data_dir
# end load_data()


def create_database(data_dir: str, batch_size: int, img_height: int, img_width: int):
    r"""
    https://www.tensorflow.org/tutorials/load_data/images#create_a_dataset
    :param data_dir:
    :param batch_size:
    :param img_height:
    :param img_width:
    :return:
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds
# end create_database()


def show_data(ds, cl):
    r"""
    https://www.tensorflow.org/tutorials/load_data/images#visualize_the_data
    :param ds: train_ds
    :param cl: class_names
    :return:
    """
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(cl[labels[i]])
            plt.axis("off")
        # end for
    # end for
    # plt.show()
# end show_data()


def standardize_data(ds):
    r"""
    https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
    :param ds: train_ds
    :return:
    """
    normalization_layer = Rescaling(1. / 255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    return image_batch, labels_batch
# end standardize_data()


def build_cnn_model(activation, input_shape):
    r"""
    https://www.tensorflow.org/tutorials/load_data/images#train_a_model
    :return:
    """
    model = Sequential()

    # standardize_data
    model.add(Rescaling(1. / 255, input_shape=input_shape))

    # 2 Convolution layer with Max polling
    model.add(Conv2D(16, 3, padding='same', activation=activation))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3, padding='same', activation=activation))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, padding='same', activation=activation))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # 2 Full connected layer
    model.add(Dense(128, activation=activation))
    model.add(Dense(5))

    # summarize the model
    model.summary()

    return model
# end build_cnn_model


def compile_and_fit_model(model, train_ds, val_ds, batch_size, n_epochs):
    # compile the model
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # define callbacks
    callbacks = [ModelCheckpoint(filepath='flower_model.h5', monitor='val_accuracy', save_best_only=True)]

    # fit the model
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_ds
    )

    return model, history
# end build_cnn_model


def show_train(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
# end show_train


def main():
    # region Load flower images
    # https://www.tensorflow.org/tutorials/load_data/images#download_the_flowers_dataset
    data_dir = load_data()

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    # endregion

    # region Create CNN database
    # https://www.tensorflow.org/tutorials/load_data/images#create_a_dataset
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds, val_ds = create_database(data_dir, batch_size, img_height, img_width)

    class_names = train_ds.class_names
    print(class_names)
    # endregion

    # region Visualize the data
    # https://www.tensorflow.org/tutorials/load_data/images#download_the_flowers_dataset
    show_data(train_ds, class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    # end for
    # endregion

    # region Configure the dataset for performance
    # https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
    autotune = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    # endregion

    # region build train model
    model = build_cnn_model(activation="relu", input_shape=(img_width, img_height, 3))
    # endregion

    # region Compile and fit
    model, history = compile_and_fit_model(model, train_ds, val_ds, batch_size, 10)

    show_train(history)
    # end
# end main()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
