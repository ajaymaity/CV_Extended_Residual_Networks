from keras import backend as K
from keras import layers
from keras import models
from keras.initializers import glorot_uniform, he_uniform, RandomUniform
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.optimizers import SGD

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def createNetwork(imgShape, classes, learningRate=0.1, activation="relu", init=he_uniform(), loss="categorical_crossentropy", do=0.0, kernel_size="same", kernel_number="same"):
    """ Copied network from GitHub -> https://github.com/keras-team/keras/blob/master/keras/applications/resnet50.py, and edited.
    Create an ExtResNet50 network.
        Args:
            imgShape: input shape of image as (image_width, image_height, image_channels),
                if keras is configured with channel_last data format; else
                (image_channel, image_width, image_height).
            classes: number of classes in the dataset.
            learningRate: the initial learning rate of the network.
            activation: the activation function of the network. One of relu/softsign/tanh. Default is relu.
            init: kernel initializer of hidden layers. One of random/glorot/he uniform. Default is he uniform.
            loss: the loss function of the network. One of categorical_crossentropy/mean_squared_error/categorical_hinge. Default is categorical_crossentropy.
            do: the dropout at the output of each block. Default is 0.0.
            kernel_size: the kernel size of the network. One of same/halved. Default is same.
            kernel_number: the number of kernels at each layer of the network. One of same/doubled. Default is same.
        Returns:
            the compiled ExtResNet50 model.
    """

    if init == "glorot-uniform":
        initFn = glorot_uniform()
    elif init == "random-uniform":
        initFn = RandomUniform()
    else:
        initFn = he_uniform()

    kernel_multiplier = 1
    if kernel_size == "halved":
        kernel_multiplier = 0.5

    kernel_no_multiplier = 1
    if kernel_number == "doubled":
        kernel_no_multiplier = 2

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

    x = Conv2D(64, (7 * kernel_no_multiplier, 7 * kernel_no_multiplier), strides=(2, 2), padding="same", name="conv1",
                kernel_initializer=initFn,
                bias_initializer="zeros")(img_input)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x1 = _conv_block(x, 3 * kernel_no_multiplier, [int(64 * kernel_multiplier), int(64 * kernel_multiplier), int(256 * kernel_multiplier)], stage=2, block="a", strides=(1, 1))
    x1 = _identity_block(x1, 3 * kernel_no_multiplier, [int(64 * kernel_multiplier), int(64 * kernel_multiplier), int(256 * kernel_multiplier)], stage=2, block="b")
    x1 = Activation(activation)(x1)
    x1 = _identity_block(x1, 3 * kernel_no_multiplier, [int(64 * kernel_multiplier), int(64 * kernel_multiplier), int(256 * kernel_multiplier)], stage=2, block="c")

    shortcut = Conv2D(int(256 * kernel_multiplier), (1, 1), strides=(1, 1),
            name="con_res2",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res2")(shortcut)
    x1 = layers.add([x1, shortcut])
    x1 = Activation(activation)(x1)
    x1 = Dropout(do, name="dropout_2")(x1)

    x2 = _conv_block(x1, 3 * kernel_no_multiplier, [int(128 * kernel_multiplier), int(128 * kernel_multiplier), int(512 * kernel_multiplier)], stage=3, block="a")
    x2 = _identity_block(x2, 3 * kernel_no_multiplier, [int(128 * kernel_multiplier), int(128 * kernel_multiplier), int(512 * kernel_multiplier)], stage=3, block="b")
    x2 = Activation(activation)(x2)
    x2 = _identity_block(x2, 3 * kernel_no_multiplier, [int(128 * kernel_multiplier), int(128 * kernel_multiplier), int(512 * kernel_multiplier)], stage=3, block="c")
    x2 = Activation(activation)(x2)
    x2 = _identity_block(x2, 3 * kernel_no_multiplier, [int(128 * kernel_multiplier), int(128 * kernel_multiplier), int(512 * kernel_multiplier)], stage=3, block="d")
    shortcut = Conv2D(int(512 * kernel_multiplier), (1, 1), strides=(2, 2),
            name="con_res3",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x1)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res3")(shortcut)
    x2 = layers.add([x2, shortcut])
    x2 = Activation(activation)(x2)
    x2 = Dropout(do, name="dropout_3")(x2)

    x3 = _conv_block(x2, 3 * kernel_no_multiplier, [int(256 * kernel_multiplier), int(256 * kernel_multiplier), int(1024 * kernel_multiplier)], stage=4, block="a")
    x3 = _identity_block(x3, 3 * kernel_no_multiplier, [int(256 * kernel_multiplier), int(256 * kernel_multiplier), int(1024 * kernel_multiplier)], stage=4, block="b")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3 * kernel_no_multiplier, [int(256 * kernel_multiplier), int(256 * kernel_multiplier), int(1024 * kernel_multiplier)], stage=4, block="c")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3 * kernel_no_multiplier, [int(256 * kernel_multiplier), int(256 * kernel_multiplier), int(1024 * kernel_multiplier)], stage=4, block="d")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3 * kernel_no_multiplier, [int(256 * kernel_multiplier), int(256 * kernel_multiplier), int(1024 * kernel_multiplier)], stage=4, block="e")
    x3 = Activation(activation)(x3)
    x3 = _identity_block(x3, 3 * kernel_no_multiplier, [int(256 * kernel_multiplier), int(256 * kernel_multiplier), int(1024 * kernel_multiplier)], stage=4, block="f")
    shortcut = Conv2D(int(1024 * kernel_multiplier), (1, 1), strides=(2, 2),
            name="con_res4",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x2)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res4")(shortcut)
    x3 = layers.add([x3, shortcut])
    x3 = Activation(activation)(x3)
    x3 = Dropout(do, name="dropout_4")(x3)

    x4 = _conv_block(x3, 3 * kernel_no_multiplier, [int(512 * kernel_multiplier), int(512 * kernel_multiplier), int(2048 * kernel_multiplier)], stage=5, block="a")
    x4 = _identity_block(x4, 3 * kernel_no_multiplier, [int(512 * kernel_multiplier), int(512 * kernel_multiplier), int(2048 * kernel_multiplier)], stage=5, block="b")
    x4 = Activation(activation)(x4)
    x4 = _identity_block(x4, 3 * kernel_no_multiplier, [int(512 * kernel_multiplier), int(512 * kernel_multiplier), int(2048 * kernel_multiplier)], stage=5, block="c")
    shortcut = Conv2D(int(2048 * kernel_multiplier), (1, 1), strides=(2, 2),
            name="con_res5",
            kernel_initializer=initFn,
            bias_initializer="zeros")(x3)
    shortcut = BatchNormalization(axis=bn_axis, name="bn_res5")(shortcut)
    x4 = layers.add([x4, shortcut])
    x4 = Activation(activation)(x4)
    x4 = Dropout(do, name="dropout_5")(x4)

    x4 = AveragePooling2D((2, 2), name="avg_pool")(x4)

    x4 = Flatten(name="flatten")(x4)

    x4 = Dense(classes, activation="softmax", name="predictions")(x4)
    model = models.Model(img_input, x4, name="ext-resnet50")

    model.compile(loss=loss,
                    optimizer=SGD(lr=learningRate, momentum=0.9, decay=0.0001),
                    metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    model.summary()
    return model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
