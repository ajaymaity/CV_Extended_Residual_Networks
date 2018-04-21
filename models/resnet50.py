from keras import backend as K
from keras import layers
from keras import models
from keras.initializers import he_uniform
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import data_utils

WEIGHTS_PATH = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5"

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def createNetwork(imgShape, classes, learningRate=0.1, pretrained=False):
    """ Copied network from GitHub -> https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py, and edited.
    Create a ResNet50 network as defined in the original paper by He et. al.
        Args:
            imgShape: input shape of image as (image_width, image_height, image_channels),
                if keras is configured with channel_last data format; else
                (image_channel, image_width, image_height).
            classes: number of classes in the dataset.
            learningRate: the initial learning rate of the network.
            pretrained: True if the network has to be pretrained using imagenet dataset.
        Returns:
            the compiled ResNet50 model.
    """

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def _conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """Copied from GitHub -> https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py, and edited.
        A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: "a","b"..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.
        # Returns
            Output tensor for the block.
        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """

        filters1, filters2, filters3 = filters
        if K.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        x = Conv2D(filters1, (1, 1), strides=strides,
                name=conv_name_base + "2a",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)

        x = Conv2D(filters2, kernel_size, padding="same",
                name=conv_name_base + "2b",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu")(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                        name=conv_name_base + "1",
                        kernel_initializer=he_uniform(),
                        bias_initializer="zeros")(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

        x = layers.add([x, shortcut])
        x = Activation("relu")(x)
        return x
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    def _identity_block(input_tensor, kernel_size, filters, stage, block):
        
        """Copied from GitHub -> https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py, and edited.
        The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: "a","b"..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        
        filters1, filters2, filters3 = filters
        if K.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        x = Conv2D(filters1, (1, 1), name=conv_name_base + "2a",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation("relu")(x)

        x = Conv2D(filters2, kernel_size,
                padding="same", name=conv_name_base + "2b",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation("relu")(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

        x = layers.add([x, input_tensor])
        x = Activation("relu")(x)
        return x
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    img_input = Input(shape=imgShape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1",
                kernel_initializer=he_uniform(),
                bias_initializer="zeros")(img_input)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = _identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    x = _conv_block(x, 3, [128, 128, 512], stage=3, block="a")
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = _identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    x = _conv_block(x, 3, [256, 256, 1024], stage=4, block="a")
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block="b")
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block="c")
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block="d")
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block="e")
    x = _identity_block(x, 3, [256, 256, 1024], stage=4, block="f")

    x = _conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = _identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    x = AveragePooling2D((2, 2), name="avg_pool")(x)

    x = Flatten(name="flatten")(x)

    if pretrained:

        x2 = Dense(1000, activation="softmax", name="pre-predictions")(x)
        model = models.Model(img_input, x2, name="pre-vgg16")
        weights_path = data_utils.get_file("resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                                    WEIGHTS_PATH,
                                    cache_subdir="models",
                                    file_hash="a7b3fe01876f51b976af0dea6bc144eb")
        model.load_weights(weights_path)

        x = Dense(classes, activation="softmax", name="predictions")(x)
        model = models.Model(img_input, x, name="resnet50")

    else:
        
        x = Dense(classes, activation="softmax", name="predictions")(x)
        model = models.Model(img_input, x, name="resnet50")

    model.compile(loss="categorical_crossentropy",
                    optimizer=SGD(lr=learningRate, momentum=0.9, decay=0.0001),
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    model.summary()
    return model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
