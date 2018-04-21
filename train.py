from __future__ import print_function #
import argparse
import datetime
import numpy as np
import os

from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from progressbar import Bar, Percentage, ProgressBar
from six.moves.urllib.request import urlretrieve
from zipfile import ZipFile

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def parseArguments():
    """ Parses the command line arguments. 
        Args: None
        Returns: The ArgumentParser object containing values of all the arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--download_data",
                        help="Set if app needs to download tiny imagenet dataset from web.",
                        action="store_true")
    parser.add_argument("--data_dir", type=str,
                        default="data/tiny-imagenet-200/tiny-imagenet-200",
                        help="Directory in which the input data is stored.")
    parser.add_argument("--reconfigure_val",
                        help="Set if val dataset should be reconfigured.",
                        action="store_true")
    parser.add_argument("--model",
                        type=str,
                        default="ext-resnet50",
                        help="Name of the architecture to train and validate the data. One of [vgg16/resnet50/ext-resnet50/ext-resnet41/ext-resnet62].")
    parser.add_argument("--pretrained",
                        help="Use imagenet pretrained model if set. Not applicable if [model] is [ext-resnet50].",
                        action="store_true")
    parser.add_argument("--name",
                        type=str,
                        default="default",
                        help="Name of this training run. Will store results in {output/log/weights}/[model]/[name].")
    parser.add_argument("--init",
                        type=str,
                        default="random-uniform",
                        help="""Kernel initializer of the hidden layers. [random-uniform/glorot-uniform/he-uniform].
                            Used only when [model] is [vgg16/ext-resnet50].""")
    parser.add_argument("--activation",
                        type=str,
                        default="relu",
                        help="""The activation function of the network. One of [relu/softsign/tanh].
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--loss",
                        type=str,
                        default="categorical_crossentropy",
                        help="""The loss function of the network. One of [categorical_crossentropy/mean_squared_error/categorical_hinge].
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--do",
                        type=float,
                        default=0.0,
                        help="""The dropout at the output of each block.
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--lr",
                        type=float,
                        default=0.1,
                        help="""The learning rate of the network.
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--kernel_size",
                        type=str,
                        default="same",
                        help="""The kernel size for the network. One of [same/halved].
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--kernel_number",
                        type=str,
                        default="same",
                        help="""The number of kernels at each layer of the network. One of [same/doubled].
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--data_aug",
                        type=str,
                        default="basic",
                        help="Data augmentation to perform. One of [basic/no/yes].")

    return parser.parse_known_args()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def reconfigureValSet(data_dir):
    """ Reconfigures the validation dataset such that each class' data is under a directory with class' name.
        Args:
            data_dir: The root directory of the images.
        Returns: None
    """
    
    # Open val_annotations.txt to find the labels for the validation images.
    with open(os.path.join(data_dir, "val", "val_annotations.txt"), "r") as text_file:
        content = text_file.readlines()
    content = [x.strip() for x in content]

    # Build dict to map image name to label.
    label_dict = {}
    for line in content:
        line_split = line.split("\t")
        label_dict[line_split[0]] = line_split[1]

    flag = False
    # Travel through the val folder.
    for subdir, dirs, files in os.walk(os.path.join(data_dir, "val")):
        for file in files:
            if file.endswith(".JPEG"):
                flag = True
                if not os.path.exists(os.path.join(data_dir, "val", label_dict[file])):
                    os.makedirs(os.path.join(data_dir, "val", label_dict[file]))
                os.rename(os.path.join(data_dir, "val", "images", file), os.path.join(data_dir, "val", label_dict[file], file))
        if flag: 
            os.rmdir(os.path.join(data_dir, "val", "images"))
            break
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def main(arguments):
    """ Main method of the file.
        Args:
            arguments: ArgumentParser object which contains user-configurable parameters.
                For more info, look into parseArguments() method.
        Returns: None
    """

    # Download data, if asked by user.
    if arguments.download_data:
        
        # Create data folder, if it doesn't exist.
        if not os.path.exists(os.path.join("data")):
            os.makedirs(os.path.join("data"))

        # Define URL to get data from,
        # and the zip folder path where the data will be stored.
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zipFolder = os.path.join("data", "tiny-imagenet-200.zip")
        
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # Define a method which will unzip the downloaded zip folder.
        def unzip():
            
            print("Unzipping data...")
            zipFile = ZipFile(zipFolder, "r")
            uncompressedSize = sum(file.file_size for file in zipFile.infolist())
            extractedSize = 0
            pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=100).start()
            start = datetime.datetime.now()
            for file in zipFile.infolist():
                extractedSize += file.file_size
                percent = extractedSize * 100 / uncompressedSize
                pbar.update(percent)
                zipFile.extract(file, path=zipFolder.rsplit(".", 1)[0])
            print("Unzipped in {} s.".format((datetime.datetime.now() - start).seconds))
            zipFile.close()
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        # Proceed to download only if the tiny imagenet folder does not already exist.
        if (not os.path.exists(os.path.join("data", "tiny-imagenet-200"))):
        
            # Proceed to download only if the downloaded zip folder does not exist,
            # else directly unzip the previously downloaded zip file.
            if (not os.path.isfile(zipFolder)):
                
                print("Retrieving dataset from web...")
                pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=100).start()
                def dlProgress(count, blockSize, totalSize):
                    percent = int(count * blockSize * 100 / totalSize)
                    pbar.update(percent)
                start = datetime.datetime.now()
                urlretrieve(url, zipFolder, reporthook=dlProgress)
                print("Downloaded in {} s.".format((datetime.datetime.now() - start).seconds))

            unzip()
            reconfigureValSet(arguments.data_dir)
        else: print("Dataset folder already exists.")

    # Define image parameters.
    imgWidth = 64 # width of image
    imgHeight = 64 # height of image
    imgChannels = 3 # channels of image, RGB
    lenTrainData = 100000 # total number of training data
    lenValData = 10000 # total number of validation data
    classes = 200 # number of classes

    # Define image shape.
    if (K.image_data_format() == "channels_first"):
        imgShape = (imgChannels, imgWidth, imgHeight)
    else:
        imgShape = (imgWidth, imgHeight, imgChannels)

    # Create Model.'
    network = arguments.model
    if (network == "vgg16"):

        batchSize = 256
        numEpochs = 2000 # 74 epochs used in the original paper
        learningRate = 0.01

        from models import vgg16
        model = vgg16.createNetwork(imgShape, classes, learningRate, arguments.pretrained, arguments.init)

    elif (network == "resnet50"):

        batchSize = 256
        numEpochs = 2000 # 60e4 iterations in the original paper
        learningRate = 0.1
        
        from models import resnet50    
        model = resnet50.createNetwork(imgShape, classes, learningRate)

    elif (network == "ext-resnet41"):

        batchSize = 256
        numEpochs = 2000
        learningRate = 0.1
        
        from models import ext_resnet41
        model = ext_resnet41.createNetwork(imgShape, classes, learningRate)
    
    elif (network == "ext-resnet50"):

        batchSize = 256
        numEpochs = 2000
        learningRate = arguments.lr
        
        from models import ext_resnet50    
        model = ext_resnet50.createNetwork(imgShape, classes, learningRate, arguments.activation, arguments.init, arguments.loss, arguments.do, arguments.kernel_size, arguments.kernel_number)
    
    elif (network == "ext-resnet62"):

        batchSize = 256
        numEpochs = 2000
        learningRate = 0.1
        
        from models import ext_resnet62    
        model = ext_resnet62.createNetwork(imgShape, classes, learningRate)
        
    # Set featurewise mean of dataset.
    if K.image_data_format() == "channels_first":
        featurewiseMean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        featurewiseStd = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    else:
        featurewiseMean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        featurewiseStd = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    # Create training data generator.
    if (arguments.data_aug == "no"):
        trainDataGen = ImageDataGenerator(featurewise_center=True)
    elif (arguments.data_aug == "yes"):
        trainDataGen = ImageDataGenerator(featurewise_center=True,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            rotation_range=20,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2)
        trainDataGen.std = featurewiseStd
    else:
        trainDataGen = ImageDataGenerator(featurewise_center=True,
                                            horizontal_flip=True,
                                            vertical_flip=True)    

    trainDataGen.mean = featurewiseMean
    trainGen = trainDataGen.flow_from_directory(os.path.join(arguments.data_dir, "train"),
                                                target_size=(imgWidth, imgHeight),
                                                batch_size=batchSize)

    # Create validation data generator.
    if arguments.reconfigure_val: reconfigureValSet(arguments.data_dir) 
    valDataGen = ImageDataGenerator(featurewise_center=True)
    valDataGen.mean = featurewiseMean
    if (arguments.data_aug == "yes"): valDataGen.std = featurewiseStd
    valGen = valDataGen.flow_from_directory(os.path.join(arguments.data_dir, "val"),
                                            target_size=(imgWidth, imgHeight),
                                            batch_size=batchSize)

    # Create directories to store output.
    if arguments.pretrained: network = network + "-imagenet"
    if not os.path.exists(os.path.join("output", network)):
        os.makedirs(os.path.join("output", network))
    name = arguments.name.replace(" ", "_")
    if not os.path.exists(os.path.join("output", network, name)):
        os.makedirs(os.path.join("output", network, name))
    outputTime = str(datetime.datetime.now()).replace(" ", "_").replace(":", ".")
    if not os.path.exists(os.path.join("output", network, name, outputTime)):
        os.makedirs(os.path.join("output", network, name, outputTime))
    if not os.path.exists(os.path.join("weights", network, name)):
        os.makedirs(os.path.join("weights", network, name))

    # Create callbacks.
    tbCallback = TensorBoard(log_dir="./logs/{}/{}/{}".format(network, name, outputTime), histogram_freq=0, write_grads=True, write_graph=True)
    lrCallback = ReduceLROnPlateau(monitor="val_categorical_accuracy", patience=5, factor=0.1, verbose=1, min_lr=0.00001)
    esCallback = EarlyStopping(monitor="val_categorical_accuracy", patience=10, verbose=1, min_delta=0.0001)

    # Fit model to the dataset...
    model.fit_generator(trainGen,
                        steps_per_epoch=lenTrainData // batchSize,
                        epochs=numEpochs,
                        validation_data=valGen,
                        validation_steps=lenValData // batchSize,
                        callbacks=[tbCallback, lrCallback, esCallback])
    
    # Save the model...
    model.save_weights(os.path.join("weights", network, name, outputTime + ".h5"))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    """ Start of program execution. """

    # Parse arguments from command line.
    args, unparsed = parseArguments()
    
    # Create folders, if they don't exist, to store results.
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("weights"):
        os.makedirs("weights")

    main(args)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
