Authors: Victor Shih, Orion Peter Fernandes

Place data files in directories like so. data dir is on the same level as the .py files. 

data/test/X_test.txt
data/test/y_test.txt
data/train/X_train.txt
data/train/y_train.txt


Training data: 561 features, 7767 samples
x_train shape = (7767, 561, 1)
y_train shape = (7767, 1)
x_test shape = (3162, 561, 1)
y_test shape = (3162, 1)

Run mainFile.py to train.

Results are stored in model_logs.