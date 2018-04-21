from keras import backend as K
from keras import layers
from keras import models
from keras.initializers import glorot_uniform, he_uniform, RandomUniform
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.optimizers import SGD

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def createNetwork(imgShape, classes, learningRate=0.1):
    """ Copied network from GitHub -> https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py, and edited.
    Create an ExtResNet-62 network.
        Args:
            imgShape: input shape of image as (image_width, image_height, image_channels),
                if keras is configured with channel_last data format; else
                (image_channel, image_width, image_height).
            classes: number of classes in the dataset.
            learningRate: the initial learning rate of the network.
        Returns:
            the compiled ExtResNet-62 model.
    """

    activation = "relu"
    initFn = he_uniform()
    lossFn = "categorical_crossentropy"

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
                kernel_initializer=initFn,
                bias_initializer="zeros")(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation(activation)(x)

        x = Conv2D(filters2, kernel_size, padding="same",
                name=conv_name_base + "2b",
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation(activation)(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c",
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                        name=conv_name_base + "1",
                        kernel_initializer=initFn,
                        bias_initializer="zeros")(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

        x = layers.add([x, shortcut])
        x = Activation(activation)(x)
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
                kernel_initializer=initFn,
                bias_initializer="zeros")(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
        x = Activation(activation)(x)

        x = Conv2D(filters2, kernel_size,
                padding="same", name=conv_name_base + "2b",
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
        x = Activation(activation)(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + "2c",
                kernel_initializer=initFn,
                bias_initializer="zeros")(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

        x = layers.add([x, input_tensor])
        # x = Activation(activation)(x)
        return x
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    img_input = Input(shape=imgShape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1",
                kernel_initializer=initFn,
                bias_initializer="zeros")(img_input)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x1 = _conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x1 = _identity_block(x1, 3, [64, 64, 256], stage=2, block="b")
    x1 = Activation(activation)(x1)
    x1 = _identity_block(x1, 3, [64, 64, 256], stage=2, block="c")
    shortcut = Conv2D(256, (1, 1), strides=(1, 1),
            name="con_res2",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res2")(shortcut)
    x1 = layers.add([x1, shortcut])
    x1 = Activation(activation)(x1)

    x2 = _conv_block(x1, 3, [128, 128, 512], stage=3, block="a")
    x2 = _identity_block(x2, 3, [128, 128, 512], stage=3, block="b")
    x2 = Activation(activation)(x2)
    x2 = _identity_block(x2, 3, [128, 128, 512], stage=3, block="c")
    x2 = Activation(activation)(x2)
    x2 = _identity_block(x2, 3, [128, 128, 512], stage=3, block="d")
    shortcut = Conv2D(512, (1, 1), strides=(2, 2),
            name="con_res3",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x1)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res3")(shortcut)
    x2 = layers.add([x2, shortcut])
    x2 = Activation(activation)(x2)

    x3 = _conv_block(x2, 3, [256, 256, 1024], stage=4, block="a")
    x3 = _identity_block(x3, 3, [256, 256, 1024], stage=4, block="b")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3, [256, 256, 1024], stage=4, block="c")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3, [256, 256, 1024], stage=4, block="d")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3, [256, 256, 1024], stage=4, block="e")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3, [256, 256, 1024], stage=4, block="f")
    shortcut = Conv2D(1024, (1, 1), strides=(2, 2),
            name="con_res4",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x2)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res4")(shortcut)
    x3 = layers.add([x3, shortcut])
    x3 = Activation(activation)(x3)

    x4 = _conv_block(x3, 3, [512, 512, 2048], stage=5, block="a")
    x4 = _identity_block(x4, 3, [512, 512, 2048], stage=5, block="b")
    x4 = Activation(activation)(x4)
    x4 = _identity_block(x4, 3, [512, 512, 2048], stage=5, block="c")
    shortcut = Conv2D(2048, (1, 1), strides=(2, 2),
            name="con_res5",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x3)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res5")(shortcut)
    x4 = layers.add([x4, shortcut])
    x4 = Activation(activation)(x4)

    x5 = _conv_block(x4, 3, [1028, 1028, 4096], stage=6, block="a")
    x5 = _identity_block(x5, 3, [1025, 1028, 4096], stage=6, block="b")
    x5 = Activation(activation)(x5)
    x5 = _identity_block(x5, 3, [1028, 1028, 4096], stage=6, block="c")
    x5 = Activation(activation)(x5)
    x5 = _identity_block(x5, 3, [1028, 1028, 4096], stage=6, block="d")
    shortcut = Conv2D(4096, (1, 1), strides=(2, 2),
            name="con_res6",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x4)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res6")(shortcut)
    x5 = layers.add([x5, shortcut])
    x5 = Activation(activation)(x5)

    x5 = Flatten(name="flatten")(x5)
    x5 = Dense(classes, activation="softmax", name="predictions")(x5)

    model = models.Model(img_input, x5, name="ext-resnet50")

    model.compile(loss=lossFn,
                    optimizer=SGD(lr=learningRate, momentum=0.9, decay=0.0001),
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    model.summary()
    return model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
