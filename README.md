# dit-dl-ca
DIT / TU Dublin: Repository to track the Deep Learning Continuous Assignment.

# Python version
```
$ python --version
Python 2.7.10

$ python -m pip --version
pip 19.1.1
```

# Dependencies
See [requirements.txt](requirements.txt)

Extra Points to note:

Epoch: 

* An epoch is one complete presentation of the data set to be learned to a learning machine.

Validation Set: 
* this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.

Mini Batchs:
* With batch size we update the weights after passing the data samples each batch
* means the gradients are calculated after passing each batch. 

sparse_softmax_cross_entropy_with_logits
* Computes sparse softmax cross entropy between logits and labels.
* Measures the probability ERROR/LOSS in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class). For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.

Learning Rate

* controls the adjust in the weights with respect to loss gradient. 

Decay Rate

* This prevents the weights from growing too large  
* When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies an exponential decay function to a provided initial learning rate. It requires a global_step value to compute the decayed learning rate.

Activation :

* It’s just a thing function that you use to get the output of node
* It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc

Relu:

* most commun activation function used on hidden layers of the neural network

Max Pooling

* The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned. 
* This is done to in part to help over-fitting by providing an abstracted form of the representation. As well, it reduces the computational cost by reducing the number of parameters to learn and provides basic translation invariance to the internal representation.
* Max pooling is done by applying a max filter to (usually) non-overlapping subregions of the initial representation.

Con2d
```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```
Depth:
* Depth of CONV layer is number of filters it is using. Depth of a filter is equal to depth of image it is using as input

