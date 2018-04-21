from keras.preprocessing import image
from keras import backend as K
from progressbar import Bar, Percentage, ProgressBar

import argparse
import numpy as np
import os

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def parseArguments():
    """ Parses the command line arguments. 
        Args: None
        Returns: The ArgumentParser object containing values of all the arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="ext-resnet50",
                        help="Name of the architecture to train and validate the data. One of [vgg16/resnet50/ext-resnet50/ext-resnet41/ext-resnet62].")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data/tiny-imagenet-200/tiny-imagenet-200/test/images",
                        help="Directory in which the test data is stored.")
    parser.add_argument("--weights",
                        type=str,
                        default="weights/ext-resnet50/he-uniform-init/2018-02-07_18.08.18.453978.h5",
                        help="Path of the weights file to load into the model.")
    parser.add_argument("--activation",
                        type=str,
                        default="relu",
                        help="""The activation function of the network. One of [relu/softsign/tanh].
                            Used only when [model] is [ext-resnet50].""")
    parser.add_argument("--do",
                        type=float,
                        default=0.0,
                        help="""The dropout at the output of each block.
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
    
    return parser.parse_known_args()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def main(arguments):
    """ This is the main method of the file.
        Args:
            arguments: ArgumentParser object which contains user-configurable parameters.
                For more info, look into parseArguments() method.
        Returns: None
    """

    imgWidth = 64
    imgHeight = 64
    imgChannels = 3
    classes = 200
    data_dir = arguments.data_dir

    # Define image shape.
    if K.image_data_format() == "channels_first":
        imgShape = (imgChannels, imgWidth, imgHeight)
        featurewiseMean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    else:
        imgShape = (imgWidth, imgHeight, imgChannels)
        featurewiseMean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    
    # Create model and load weights.
    network = arguments.model
    weights_path = arguments.weights
    if (network == "ext-resnet50"):
        from models import ext_resnet50
        model = ext_resnet50.createNetwork(imgShape, classes, activation=arguments.activation, do=arguments.do, kernel_size=arguments.kernel_size, kernel_number=arguments.kernel_number)
        model.load_weights(weights_path)
    elif (network == "resnet50"):
        from models import resnet50
        model = resnet50.createNetwork(imgShape, classes)
        model.load_weights(weights_path)
    elif (network == "vgg16"):
        from models import vgg16
        model = vgg16.createNetwork(imgShape, classes)
        model.load_weights(weights_path)

    # Store the classes/indices in a dict.
    class_index = {"n01910747": 13, "n02730930": 62, "n07873807": 191, "n02123394": 31, "n04417672": 162, "n02124075": 32, "n02190166": 37, "n03854065": 126, "n04265275": 151, "n09193705": 194, "n07695742": 181, "n02927161": 82, "n04275548": 152, "n02788148": 64, "n03085013": 92, "n07749582": 187, "n02206856": 38, "n04074963": 141, "n03089624": 93, "n03670208": 116, "n03100240": 94, "n02395406": 48, "n02814860": 71, "n04328186": 155, "n02504458": 57, "n01644900": 3, "n02963159": 85, "n07720875": 184, "n02279972": 44, "n01917289": 14, "n03042490": 91, "n04562935": 173, "n09332890": 197, "n02099601": 26, "n03983396": 135, "n09428293": 198, "n03891332": 127, "n01882714": 12, "n02843684": 76, "n02509815": 58, "n02321529": 46, "n04532670": 170, "n02437312": 53, "n09246464": 195, "n02125311": 33, "n04560804": 172, "n01768244": 6, "n03837869": 124, "n02906734": 79, "n03447447": 108, "n02410509": 50, "n03014705": 89, "n01984695": 19, "n04507155": 168, "n07715103": 183, "n04146614": 145, "n04285008": 153, "n04540053": 171, "n07753592": 188, "n03584254": 110, "n02999410": 88, "n02808440": 69, "n02795169": 67, "n02669723": 60, "n03544143": 109, "n01770393": 7, "n03970156": 131, "n03930313": 129, "n02480495": 54, "n03770439": 120, "n04532106": 169, "n02769748": 63, "n07871810": 190, "n01774750": 9, "n06596364": 176, "n04118538": 143, "n02403003": 49, "n07614500": 179, "n02085620": 24, "n04008634": 137, "n04133789": 144, "n02841315": 75, "n02165456": 36, "n07579787": 177, "n03937543": 130, "n02977058": 86, "n02481823": 55, "n01698640": 4, "n04487081": 166, "n03404251": 105, "n04376876": 159, "n02226429": 39, "n04371430": 158, "n03804744": 122, "n03424325": 106, "n04067472": 139, "n02058221": 22, "n03662601": 115, "n02948072": 83, "n02793495": 66, "n01944390": 15, "n02106662": 28, "n04149813": 146, "n03355925": 101, "n02823428": 73, "n01855672": 11, "n02815834": 72, "n01629819": 1, "n03814639": 123, "n03599486": 111, "n03977966": 133, "n02415577": 51, "n02666196": 59, "n02233338": 41, "n02837789": 74, "n02236044": 42, "n01784675": 10, "n07747607": 186, "n03763968": 119, "n04099969": 142, "n02791270": 65, "n02883205": 77, "n04597913": 175, "n04259630": 150, "n04254777": 149, "n04311004": 154, "n02892201": 78, "n02917067": 81, "n04356056": 156, "n02056570": 21, "n03026506": 90, "n07711569": 182, "n04366367": 157, "n02094433": 25, "n02002724": 20, "n03796401": 121, "n09256479": 196, "n02099712": 27, "n01774384": 8, "n02486410": 56, "n04486054": 165, "n02268443": 43, "n07734744": 185, "n04023962": 138, "n02281406": 45, "n03393912": 103, "n03179701": 97, "n02699494": 61, "n12267677": 199, "n07615774": 180, "n03976657": 132, "n04465501": 164, "n03160309": 96, "n03733131": 118, "n03126707": 95, "n02074367": 23, "n02909870": 80, "n02123045": 30, "n04398044": 160, "n02132136": 35, "n04596742": 174, "n07768694": 189, "n02423022": 52, "n03201208": 98, "n02802426": 68, "n03838899": 125, "n01641577": 2, "n07920052": 193, "n03706229": 117, "n01443537": 0, "n01983481": 18, "n07875152": 192, "n03617480": 112, "n03250847": 99, "n03255030": 100, "n02231487": 40, "n02814533": 70, "n02113799": 29, "n04251144": 148, "n04179913": 147, "n03444034": 107, "n03902125": 128, "n07583066": 178, "n03992509": 136, "n04501370": 167, "n03980874": 134, "n03649909": 114, "n01742172": 5, "n04456115": 163, "n02950826": 84, "n03400231": 104, "n03388043": 102, "n01950731": 17, "n04070727": 140, "n02364673": 47, "n01945685": 16, "n04399382": 161, "n02129165": 34, "n03637318": 113, "n02988304": 87}
    index_class = dict()
    for classes in class_index.keys():
        index_class[class_index[classes]] = classes

    i = 0
    data = "imid,n01443537,n01629819,n01641577,n01644900,n01698640,n01742172,n01768244,n01770393,n01774384,n01774750,n01784675,n01855672,n01882714,n01910747,n01917289,n01944390,n01945685,n01950731,n01983481,n01984695,n02002724,n02056570,n02058221,n02074367,n02085620,n02094433,n02099601,n02099712,n02106662,n02113799,n02123045,n02123394,n02124075,n02125311,n02129165,n02132136,n02165456,n02190166,n02206856,n02226429,n02231487,n02233338,n02236044,n02268443,n02279972,n02281406,n02321529,n02364673,n02395406,n02403003,n02410509,n02415577,n02423022,n02437312,n02480495,n02481823,n02486410,n02504458,n02509815,n02666196,n02669723,n02699494,n02730930,n02769748,n02788148,n02791270,n02793495,n02795169,n02802426,n02808440,n02814533,n02814860,n02815834,n02823428,n02837789,n02841315,n02843684,n02883205,n02892201,n02906734,n02909870,n02917067,n02927161,n02948072,n02950826,n02963159,n02977058,n02988304,n02999410,n03014705,n03026506,n03042490,n03085013,n03089624,n03100240,n03126707,n03160309,n03179701,n03201208,n03250847,n03255030,n03355925,n03388043,n03393912,n03400231,n03404251,n03424325,n03444034,n03447447,n03544143,n03584254,n03599486,n03617480,n03637318,n03649909,n03662601,n03670208,n03706229,n03733131,n03763968,n03770439,n03796401,n03804744,n03814639,n03837869,n03838899,n03854065,n03891332,n03902125,n03930313,n03937543,n03970156,n03976657,n03977966,n03980874,n03983396,n03992509,n04008634,n04023962,n04067472,n04070727,n04074963,n04099969,n04118538,n04133789,n04146614,n04149813,n04179913,n04251144,n04254777,n04259630,n04265275,n04275548,n04285008,n04311004,n04328186,n04356056,n04366367,n04371430,n04376876,n04398044,n04399382,n04417672,n04456115,n04465501,n04486054,n04487081,n04501370,n04507155,n04532106,n04532670,n04540053,n04560804,n04562935,n04596742,n04597913,n06596364,n07579787,n07583066,n07614500,n07615774,n07695742,n07711569,n07715103,n07720875,n07734744,n07747607,n07749582,n07753592,n07768694,n07871810,n07873807,n07875152,n07920052,n09193705,n09246464,n09256479,n09332890,n09428293,n12267677"
    dataList = data.split(",")
    doc = data + "\n"
    pbar = ProgressBar(widgets=[Bar()], maxval=10000).start()
    # Run model over all images
    for filename in os.listdir(data_dir):
        
        img = image.load_img(os.path.join(data_dir, filename), target_size=(imgWidth, imgHeight))
        imgArr = image.img_to_array(img)
        trImg = imgArr - featurewiseMean    
        trImg = trImg.reshape(1, imgShape[0], imgShape[1], imgShape[2])
        prediction = model.predict(trImg)
    
        for clss in dataList:
            
            if (clss == "imid"): doc += str(i) + ","
            else:
                idx = class_index[clss]
                doc += str(prediction[0][idx]) + ","
        doc = doc[:-1] + "\n"

        pbar.update(i + 1)
        i += 1

    # Create folder if it does not exist.
    if not os.path.exists("results"):
        os.makedirs("results")

    with open(os.path.join("results", network + ".txt"), "a") as resultsFile:
        resultsFile.write(doc)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    """ Start of program execution. """

    args, unparsed = parseArguments()
    main(args)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
