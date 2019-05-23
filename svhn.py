import svhn_preprocessing
import svhn_train
import svhn_test

# This function when called should (a) download the training and test data, (b) train the model from scratch; and (c)
# perform analysis against test data. The final output of this function should be a production of average F1 scores
# across each class in your testset.
# Must return f1-score
def traintest():
    ret = svhn_preprocessing.run()
    ret += svhn_train.run()
    return ret


# which takes the name of a JPEG or PNG file that is assumed to be the same dimensions as the standard SVHN test data
# and return an integer that corresponds to the most likely house number seen in the supplied image. This result
# should be produced according to a tensorflow model that you have pre-trained and included in the archive.
# Expect either PNG and JPG
# result must be returned (Not only printed)
def test(img_path):
    return svhn_test.classify(img_path)
