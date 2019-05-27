# C-NN & Street View House Numbers Dataset (SVHN)
DIT / TU Dublin: Repository to track the Deep Learning Continuous Assignment.

## Contributors

- Bruno Ambrozio | Class: DT228B | bruno.ambrozio@mydit.ie
- Victor Zacchi | Student ID: D16128783 | Class: DT228B/ASD | victor.zacchi@mydit.ie

## Lecturer
 - Robert Ross | Deep Learning | Technological University Dublin | Dublin, Ireland


## Environment information

### Hardware where the model was trained
```
$ uname -a
Darwin Brunos-MBP 18.6.0 Darwin Kernel Version 18.6.0: Thu Apr 25 23:16:27 PDT 2019; root:xnu-4903.261.4~2/RELEASE_X86_64 x86_64
$ sw_vers
ProductName:    Mac OS X
ProductVersion: 10.14.5
BuildVersion:   18F132

$ system_profiler SPHardwareDataType
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: MacBookPro14,3
      Processor Name: Intel Core i7
      Processor Speed: 2.9 GHz
      Number of Processors: 1
      Total Number of Cores: 4
      L2 Cache (per Core): 256 KB
      L3 Cache: 8 MB
      Hyper-Threading Technology: Enabled
      Memory: 16 GB
      Boot ROM Version: 194.0.0.0.0
      SMC Version (system): 2.45f0
```

### Python, pip and library versions
```
$ python --version
Python 2.7.10

$ python -m pip --version
pip 19.1.1 from /Users/bambrozi/workspace/github.com/bambrozio/dit-dl-ca/env/lib/python2.7/site-packages/pip (python 2.7)

$ python -m pip freeze
absl-py==0.7.1
astor==0.8.0
backports.functools-lru-cache==1.5
backports.weakref==1.0.post1
cycler==0.10.0
enum34==1.1.6
funcsigs==1.0.2
futures==3.2.0
gast==0.2.2
grpcio==1.20.1
h5py==2.9.0
Keras-Applications==1.0.7
Keras-Preprocessing==1.0.9
kiwisolver==1.1.0
Markdown==3.1.1
matplotlib==2.2.4
mock==3.0.5
numpy==1.16.3
Pillow==6.0.0
protobuf==3.7.1
pyparsing==2.4.0
python-dateutil==2.8.0
pytz==2019.1
scikit-learn==0.20.3
scipy==1.2.1
six==1.12.0
sklearn==0.0
subprocess32==3.5.4
tensorboard==1.13.1
tensorflow==1.13.1
tensorflow-estimator==1.13.0
termcolor==1.1.0
Werkzeug==0.15.4
```

## Source Code Reference
Adapted from: 
1. B. Diesel, Decoding a Sequences of Digits in Real World Photos Using a Convolutional Neural Network. 2016 [Online]. Available: https://github.com/bdiesel/tensorflow-svhn. [Accessed: 27- May- 2019]

## References
1. F. Adelkhani, Understanding Convolutional Neural Networks + Edge Detectors, ConvNets, and DIGITS. 2019 [Online]. Available: https://www.youtube.com/watch?v=SbLJLT8xXD0. [Accessed: 27- May- 2019]
1. T. Almenningen, "05-svhn-multi-preprocessing.ipynb", GitHub, 2017. [Online]. Available: https://github.com/thomalm/svhn-multi-digit/blob/master/05-svhn-multi-preprocessing.ipynb. [Accessed: 27- May- 2019]
1. T. Almenningen, "06-svhn-multi-model.ipynb", GitHub, 2017. [Online]. Available: https://github.com/thomalm/svhn-multi-digit/blob/master/06-svhn-multi-model.ipynb. [Accessed: 27- May- 2019]
1. K. Chuang, "svhn-preprocessing.ipynb", GitHub, 2018. [Online]. Available: https://github.com/k-chuang/tf-svhn/blob/master/svhn-preprocessing.ipynb. [Accessed: 27- May- 2019]
1. K. Chuang, "svhn-model.ipynb", GitHub, 2018. [Online]. Available: https://github.com/k-chuang/tf-svhn/blob/master/svhn-model.ipynb. [Accessed: 27- May- 2019]
1. "Convolutional Neural Network", Mathworks.com, 2019. [Online]. Available: https://www.mathworks.com/solutions/deep-learning/convolutional-neural-network.html. [Accessed: 27- May- 2019]
1. B. Diesel, Decoding a Sequences of Digits in Real World Photos Using a Convolutional Neural Network. 2016 [Online]. Available: https://github.com/bdiesel/tensorflow-svhn/blob/master/capstone%20report%20final.pdf. [Accessed: 27- May- 2019]
1. M. Levoy, K. Dektar and A. Adams, "Spatial convolution", Graphics.stanford.edu, 2012. [Online]. Available: https://graphics.stanford.edu/courses/cs178/applets/convolution.html. [Accessed: 27- May- 2019]
1. Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu and A. Y. Ng, "Reading Digits in Natural Images with Unsupervised Feature Learning", Ufldl.stanford.edu, 2011. [Online]. Available: http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf. [Accessed: 27- May- 2019]
1. Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu and A. Y. Ng, "The Street View House Numbers (SVHN) Dataset", Ufldl.stanford.edu, 2011. [Online]. Available: http://ufldl.stanford.edu/housenumbers/. [Accessed: 27- May- 2019]
