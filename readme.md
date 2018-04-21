Arguments:
download_data #
data_dir #
reconfigure_val
model #
pretrained #
name
init #

This program can be run on python 2 or 3.

### Set up for the  first time
pip install -r requirements.txt

## TRAINING
### Run program for the first time
python train.py --download_data

This will download the data from web, and start running my implementation, ExtResNet-50, with all the default parameters.

Once the data is downloaded in data/tiny-imagenet-200/tiny-imagenet-200/, every subsequent execution can be performed using the command:
python train.py

### Command line arguments
usage: train.py [-h] [--download_data] [--data_dir DATA_DIR]
                [--reconfigure_val] [--model MODEL] [--pretrained]
                [--name NAME] [--init INIT] [--activation ACTIVATION]
                [--loss LOSS] [--do DO] [--lr LR] [--kernel_size KERNEL_SIZE]
                [--kernel_number KERNEL_NUMBER] [--data_aug DATA_AUG]

optional arguments:
  -h, --help            show this help message and exit
  --download_data       Set if app needs to download tiny imagenet dataset
                        from web.
  --data_dir DATA_DIR   Directory in which the input data is stored.
  --reconfigure_val     Set if val dataset should be reconfigured.
  --model MODEL         Name of the architecture to train and validate the
                        data. One of [vgg16/resnet50/ext-resnet50
  --pretrained          Use imagenet pretrained model if set. Not applicable
                        if model = ext-resnet50.
  --name NAME           Name of this training run. Will store results in
                        {output/log/weights}/[model]/[name].
  --init INIT           Kernel initializer of the hidden layers. [random-
                        uniform/glorot-uniform/he-uniform]. Used only when
                        [model] is [vgg16/ext-resnet50].
  --activation ACTIVATION
                        The activation function of the network. One of
                        [relu/softsign/tanh]. Used only when [model] is [ext-
                        resnet50].
  --loss LOSS           The loss function of the network. One of [categorical_
                        crossentropy/mean_squared_error/categorical_hinge].
                        Used only when [model] is [ext-resnet50].
  --do DO               The dropout at the output of each block. Used only
                        when [model] is [ext-resnet50].
  --lr LR               The learning rate of the network. Used only when
                        [model] is [ext-resnet50].
  --kernel_size KERNEL_SIZE
                        The kernel size for the network. One of [same/halved].
                        Used only when [model] is [ext-resnet50].
  --kernel_number KERNEL_NUMBER
                        The number of kernels at each layer of the network.
                        One of [same/doubled]. Used only when [model] is [ext-
                        resnet50].
  --data_aug DATA_AUG   Data augmentation to perform. One of [basic/no/yes].

### To run ExtResNet-62
python train.py --model ext-resnet62

### To run VGG-16 with pretrained weights
python train.py --model vgg16 --pretrained

## PREDICTION
To predict on a test data set, type the following command:
python predict.py

### Command line arguments
usage: predict.py [-h] [--model MODEL] [--data_dir DATA_DIR]
                  [--weights WEIGHTS] [--activation ACTIVATION] [--do DO]
                  [--kernel_size KERNEL_SIZE] [--kernel_number KERNEL_NUMBER]
                  [--data_aug DATA_AUG]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name of the architecture to train and validate the
                        data. One of [vgg16/resnet50/ext-resnet50/ext-resnet41
                        /ext-resnet62].
  --data_dir DATA_DIR   Directory in which the test data is stored.
  --weights WEIGHTS     Path of the weights file to load into the model.
  --activation ACTIVATION
                        The activation function of the network. One of
                        [relu/softsign/tanh]. Used only when [model] is [ext-
                        resnet50].
  --do DO               The dropout at the output of each block. Used only
                        when [model] is [ext-resnet50].
  --kernel_size KERNEL_SIZE
                        The kernel size for the network. One of [same/halved].
                        Used only when [model] is [ext-resnet50].
  --kernel_number KERNEL_NUMBER
                        The number of kernels at each layer of the network.
                        One of [same/doubled]. Used only when [model] is [ext-
                        resnet50].
  --data_aug DATA_AUG   Data augmentation to perform. One of [basic/no/yes].

## CONFUSION MATRIX
To plot confusion matrix on a validation data set, type the following command:
python conf_matrix.py

### Command line arguments
usage: conf_matrix.py [-h] [--num_classes NUM_CLASSES] [--model MODEL]
                      [--data_dir DATA_DIR] [--weights WEIGHTS] [--name NAME]
                      [--activation ACTIVATION] [--do DO]
                      [--kernel_size KERNEL_SIZE]
                      [--kernel_number KERNEL_NUMBER]

optional arguments:
  -h, --help            show this help message and exit
  --num_classes NUM_CLASSES
                        Number of classes
  --model MODEL         Name of the architecture to train and validate the
                        data. One of [vgg16/resnet50/ext-resnet50/ext-resnet41
                        /ext-resnet62].
  --data_dir DATA_DIR   Directory in which the val data is stored.
  --weights WEIGHTS     Path of the weights file to load into the model.
  --name NAME           Name of the confusion matrix and image file
  --activation ACTIVATION
                        The activation function of the network. One of
                        [relu/softsign/tanh]. Used only when [model] is [ext-
                        resnet50].
  --do DO               The dropout at the output of each block. Used only
                        when [model] is [ext-resnet50].
  --kernel_size KERNEL_SIZE
                        The kernel size for the network. One of [same/halved].
                        Used only when [model] is [ext-resnet50].
  --kernel_number KERNEL_NUMBER
                        The number of kernels at each layer of the network.
                        One of [same/doubled]. Used only when [model] is [ext-
                        resnet50].