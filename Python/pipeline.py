from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from model_0 import model

MODEL_NAME = 'model_0'

### HYPERPARAMETERS
BATCH_SIZE = 32
EPOCHS = 10

### CIFAR10 dataset loading:
### Partition data - data is already partioned from unpacking here:
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

input_shape = (32,32,3) # get 1st sample's shape.

# Check shape of each partition. Each img is 32x32x3. 50000 in training set, 10000 in test set.
print("x_train shape = " + str(np.shape(x_train)))
print("y_train shape = " + str(np.shape(y_train)))
print("x_test shape = " + str(np.shape(x_test)))
print("y_test shape = " + str(np.shape(y_test)))
###================================================================================================
### We are using model in the model_0.py file. Change this to load other models.
from model_0 import model

### Plotting function
def plot_Acc_And_Loss(history_dict, save=True):

    #Plots loss and accuracy of train and val data over epochs. @@ -55,16 +41,38 @@ def plot_Acc_And_Loss(history_dict, save=True):
plt.show()
###================================================================================================
print("Running...")
# Specify model name to save model as. eg., "model_0", "model_1", "model_2"
MODEL_NAME = 'model_0'

###================================================================================================
BATCH_SIZE = 32
EPOCHS = 10

### CIFAR10 dataset loading:
### Partition data - data is already partioned from unpacking here:
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
input_shape = (32,32,3) # get 1st sample's shape.

# Check shape of each partition. Each img is 32x32x3. 50000 in training set, 10000 in test set.
print("x_train shape = " + str(np.shape(x_train)))
print("y_train shape = " + str(np.shape(y_train)))
print("x_test shape = " + str(np.shape(x_test)))
print("y_test shape = " + str(np.shape(y_test)))


###================================================================================================
### Compile a model.
model = model(input_shape)
opt = Adam(learning_rate=.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics=['accuracy']
model.compile(optimizer=opt, loss=loss, metrics=metrics)
model.summary()

###================================================================================================
### Train and Predict.
model_checkpoint = ModelCheckpoint(filepath='model/'+MODEL_NAME,
                                       verbose=1,
@@ -75,7 +83,12 @@ def plot_Acc_And_Loss(history_dict, save=True):
# t0 = len(x_train)//BATCH_SIZE
model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, callbacks=[csv_logger, model_checkpoint], validation_data=(x_test, y_test))


###================================================================================================
"""Save model history and plot loss and acc"""
"""
Note!!! If model history 
"""
with open('model/'+MODEL_NAME+'/trainHistoryDict', 'wb') as file_name:
    pickle.dump(model_history.history, file_name)       # Save history dict
plot_Acc_And_Loss(model_history.history)        # Plot acc and loss over epochs
@@ -84,9 +97,15 @@ def plot_Acc_And_Loss(history_dict, save=True):
    model.summary(print_fn=lambda x: fh.write(x + '\n'))


###================================================================================================
### Evaluate model.
print("\nEvaluating model...\n")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

pred_outs = model.predict(x_test)

pred_labels = np.argmax(pred_outs,axis=1)
pred_labels = np.argmax(pred_outs,axis=1)


# t0model = load_model("model/model_0") # Load a saved model from "model/..." and evaluate.
# t0predict = t0model.evaluate(x_test,  y_test, verbose=2)
