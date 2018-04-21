from keras import models
from keras import regularizers
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform, RandomUniform
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import data_utils

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def createNetwork(imgShape, classes, learningRate=0.01, pretrained=False, init=RandomUniform()):
    """ Create a VGG16 network as defined in the original paper by Simonyan and Zisserman.
        Args:
            imgShape: input shape of image as (image_width, image_height, image_channels),
                if keras is configured with channel_last data format; else
                (image_channel, image_width, image_height).
            classes: number of classes in the dataset.
            learningRate: the initial learning rate of the network.
            pretrained: True if the network has to be pretrained using imagenet dataset.
            init: kernel initializer of hidden layers. One of random/glorot/he uniform. Default is random uniform.
        Returns:
            the compiled VGG model.
    """

    if init == "glorot-uniform":
        initFn = glorot_uniform()
    elif init == "he-uniform":
        initFn = he_uniform()
    else:
        initFn = RandomUniform()

    img_input = Input(shape=imgShape)

    # Block 1
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(img_input)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3",
                kernel_regularizer=regularizers.l2(5e-4),
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

    x = Flatten(name="flatten")(x)

    if pretrained:

        model = models.Model(img_input, x, name="pre-vgg16")
        weights_path = data_utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                            WEIGHTS_PATH,
                                            cache_subdir='models',
                                            file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)

        x = Dense(4096, activation='relu', name='fc1',
                    kernel_regularizer=regularizers.l2(5e-4),
                    kernel_initializer=initFn,
                    bias_initializer='zeros')(x)
        x = Dropout(0.5, name='dropout_fc1')(x)        
        x = Dense(4096, activation='relu', name='fc2',
                    kernel_regularizer=regularizers.l2(5e-4),
                    kernel_initializer=initFn,
                    bias_initializer='zeros')(x)
        x = Dropout(0.5, name='dropout_fc2')(x)
        x = Dense(classes, activation="softmax", name="predictions")(x)
        model = models.Model(img_input, x, name="vgg16")

    else:

        x = Dense(classes, activation="softmax", name="predictions")(x)
        model = models.Model(img_input, x, name="vgg16")

    model.compile(loss="categorical_crossentropy",
                    optimizer=SGD(lr=learningRate, momentum=0.9),
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    model.summary()
    return model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
