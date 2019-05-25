# dit-dl-ca
DIT / TU Dublin: Repository to track the Deep Learning Continuous Assignment.

# Dependencies

python -m pip install \
  tensorflow==1.13.1 \
  h5py \
  scipy \
  sklearn \
  Pillow

Extra Points to note:

Epoch: 
#An epoch is one complete presentation of the data set to be learned to a learning machine.

Validation Set: 
# this data set is used to minimize overfitting. You're not adjusting the weights of the network with this data set, you're just verifying that any increase in accuracy over the training data set actually yields an increase in accuracy over a data set that has not been shown to the network before, or at least the network hasn't trained on it (i.e. validation data set). If the accuracy over the training data set increases, but the accuracy over the validation data set stays the same or decreases, then you're overfitting your neural network and you should stop training.

Mini Batchs:
# With batch size we update the weights after passing the data samples each batch
# means the gradients are calculated after passing each batch. 

sparse_softmax_cross_entropy_with_logits
# Computes sparse softmax cross entropy between logits and labels.
# Measures the probability ERROR/LOSS in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class). For example, each CIFAR-10 image is labeled with one and only one label: an image can be a dog or a truck, but not both.

Learning Rate
# controls the adjust in the weights with respect to loss gradient. 

Decay Rate
# This prevents the weights from growing too large  
# When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies an exponential decay function to a provided initial learning rate. It requires a global_step value to compute the decayed learning rate.
