import os

# This function when called should (a) download the training and test data, (b) train the model from scratch; and (c)
# perform analysis against test data. The final output of this function should be a production of average F1 scores
# across each class in your testset.
# Must return f1-score
def traintest():
    os.system('echo "" > harness.traintest.log')
    os.system('python svhn_preprocessing.py >> harness.traintest.log')
    os.system('python svhn_train_classifier.py >> harness.traintest.log')
    os.system('python svhn_train_regressor.py >> harness.traintest.log')


# which takes the name of a JPEG or PNG file that is assumed to be the same dimensions as the standard SVHN test data
# and return an integer that corresponds to the most likely house number seen in the supplied image. This result
# should be produced according to a tensorflow model that you have pre-trained and included in the archive.
# Expect either PNG and JPG
# result must be returned (Not only printed)
def test(img_path):
    os.system('python svhn_test.py {}'.format(img_path))
