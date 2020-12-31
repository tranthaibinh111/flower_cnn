# region Python
import os
import pathlib
# endregion

# region NumPy
import numpy as np
# endregion

# region TensorFlow
import tensorflow as tf
# import keras (high level API) with tensorflow as backend
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3 as GoogLeNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
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


def standardize_data(ds, input_shape):
    r"""
    https://www.tensorflow.org/tutorials/load_data/images#standardize_the_data
    :param ds: train_ds
    :param input_shape:
    :return:
    """
    normalization_layer = Rescaling(1. / 255)

    if input_shape:
        ds = ds.map(lambda x, y: (tf.image.resize(x, size=input_shape), y))

    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))

    return normalized_ds
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


def build_alexnet_model(input_shape):
    model = Sequential()

    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Layer: Conv (w ReLu)
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(BatchNormalization())

    # 4th Layer: Conv (w ReLu) splitted into two groups
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(BatchNormalization())

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    model.add(Flatten())

    # 7th Layer: FC (w ReLu) -> Dropout
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 8th Layer: FC (w ReLu) -> Dropout
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation="softmax"))

    # summarize the model
    model.summary()

    return model
# end build_cnn_model


def build_vgg16_model(input_shape):
    vgg16_model = VGG16(include_top=False, weights=None, input_shape=input_shape)

    x = vgg16_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(.5, name='fc1_drop')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(.5, name='fc2_drop')(x)
    predictions = Dense(5, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=vgg16_model.input, outputs=predictions)
    # summarize the model
    model.summary()

    return model
# end build_vgg16_model


def build_googlenet_model(input_shape):
    goolenet_model = GoogLeNet(include_top=False, weights=None, input_shape=input_shape)

    x = goolenet_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(.5, name='fc1_drop')(x)
    predictions = Dense(5, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=goolenet_model.input, outputs=predictions)
    # summarize the model
    model.summary()

    return model
    # return GoogLeNet(weights=None, input_shape=input_shape, classes=5)
# end build_goolenet_model


def build_resnet50_model(input_shape):
    resnet50_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)

    x = resnet50_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(.5, name='fc1_drop')(x)
    predictions = Dense(5, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=resnet50_model.input, outputs=predictions)
    # summarize the model
    model.summary()

    return model
# end build_resnet50_model


def build_mobilenetv2_model(input_shape):
    mobilenetv2_model = MobileNetV2(include_top=False, weights=None, input_shape=input_shape)

    x = mobilenetv2_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(.5, name='fc1_drop')(x)
    predictions = Dense(5, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=mobilenetv2_model.input, outputs=predictions)
    # summarize the model
    model.summary()

    return model
# end build_mobilenetv2_model


def compile_and_fit_model(model_name, model, train_ds, val_ds, batch_size, n_epochs):
    # compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=10e-4),
        # optimizer=keras.optimizers.RMSprop(learning_rate=10e-6),
        # optimizer=keras.optimizers.SGD(learning_rate=10e-4, momentum=0.9, nesterov=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

    # define callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f'models/{model_name}/flower_adam_10e-4_model_202101010159.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

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


def show_train(history, show=False):
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    if show:
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
    batch_size = 4
    img_height = 180
    img_width = 180
    n_epoch = 50

    train_ds, val_ds = create_database(data_dir, batch_size, img_height, img_width)

    class_names = train_ds.class_names
    print(class_names)
    # endregion

    # # region Visualize the data
    # # https://www.tensorflow.org/tutorials/load_data/images#download_the_flowers_dataset
    # show_data(train_ds, class_names)
    #
    # for image_batch, labels_batch in train_ds:
    #     print(image_batch.shape)
    #     print(labels_batch.shape)
    #     break
    # # end for
    # # endregion

    # region Configure the dataset for performance
    # https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
    # autotune = tf.data.experimental.AUTOTUNE
    #
    # train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    # val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    # endregion

    # region CNN Model
    # cnn_model = build_cnn_model(activation="relu", input_shape=(img_width, img_height, 3))
    #
    # cnn_model, cnn_history = compile_and_fit_model(
    #     'simple',
    #     cnn_model,
    #     standardize_data(train_ds),
    #     standardize_data(val_ds),
    #     batch_size,
    #     n_epoch
    # )
    #
    # show_train(cnn_history)
    # endregion

    # region AlexNet Model
    alexnet_model = build_alexnet_model(input_shape=(227, 227, 3))

    alexnet_model, alexnet_history = compile_and_fit_model(
        'alexnet',
        alexnet_model,
        standardize_data(train_ds, (227, 227)),
        standardize_data(val_ds, (227, 227)),
        batch_size,
        n_epoch
    )

    show_train(alexnet_history)
    # endregion

    # region GooleNet Model
    goolenet_model = build_googlenet_model(input_shape=(224, 224, 3))

    goolenet_model, goolenet_history = compile_and_fit_model(
        'googlenet',
        goolenet_model,
        standardize_data(train_ds, (224, 224)),
        standardize_data(val_ds, (224, 224)),
        batch_size,
        n_epoch
    )

    show_train(goolenet_history)
    # endregion

    # region ResNet50 Model
    resnet50_model = build_resnet50_model(input_shape=(224, 224, 3))

    resnet50_model, resnet50_history = compile_and_fit_model(
        'resnet50',
        resnet50_model,
        standardize_data(train_ds, (224, 224)),
        standardize_data(val_ds, (224, 224)),
        batch_size,
        n_epoch
    )

    show_train(resnet50_history)
    # endregion

    # region MobileNet V2 Model
    mobilenetv2_model = build_mobilenetv2_model(input_shape=(224, 224, 3))

    mobilenetv2_model, mobilenetv2_history = compile_and_fit_model(
        'mobilenetv2',
        mobilenetv2_model,
        standardize_data(train_ds, (224, 224)),
        standardize_data(val_ds, (224, 224)),
        batch_size,
        n_epoch
    )

    show_train(mobilenetv2_history, show=True)
    # endregion

    # region build train model
    # model = build_vgg16_model(input_shape=(img_width, img_height, 3))
    # endregion

    # model = load_model('flower_vgg16_model.h5')
    #
    # for image_batch, labels_batch in val_ds:
    #     for index in range(batch_size - 1):
    #         try:
    #             print(f'y_true: {class_names[labels_batch[index]]}')
    #             # convert the image pixels to a numpy array
    #             image = image_batch[index]
    #             # convert the image pixels to a numpy array
    #             image = image.numpy()
    #             # reshape data for the model
    #             image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #             predict = model.predict_classes(image, batch_size=1)
    #             print(predict)
    #             print(f'y_predict: {class_names[predict[0]]}')
    #             print('-----------------------------------------------------------')
    #         except Exception as e:
    #             print(e)

# end main()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
