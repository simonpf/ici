from explore         import IciData
from keras.models    import Sequential
from keras.layers    import Dense, Activation
from sklearn.metrics import roc_curve
from pca             import PCA

import matplotlib.pyplot as plt
import numpy as np

training_data = IciData('/home/simonpf/projects/ici/data/sets/full/train.nc')
test_data     = IciData('/home/simonpf/projects/ici/data/sets/full/test.nc')

# Load data and perform SVD
x_train     = training_data.get_input_data()
x_test      = test_data.get_input_data()
u,s,v       = np.linalg.svd(x_train, full_matrices=0)
pca         = PCA.fromRMatrix(v)
x_train_pca = pca.apply(x_train)
x_test_pca  = pca.apply(x_test)
y_train     = training_data.get_output_data("clear_sky")
y_test      = test_data.get_output_data("clear_sky")

## Set up train deep models.
#model_deep = Sequential()
#model_deep.add(Dense(input_dim = 11, units = 32))
#model_deep.add(Activation('relu'))
#model_deep.add(Dense( units = 32))
#model_deep.add(Activation('relu'))
#model_deep.add(Dense( units = 32))
#model_deep.add(Activation('relu'))
#model_deep.add(Dense( units = 32))
#model_deep.add(Activation('relu'))
#model_deep.add(Dense( units = 32))
#model_deep.add(Dense(units = 1))
#model_deep.add(Activation('sigmoid'))
#
#model_deep.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
#
#model_deep.fit(x_train, y_train, epochs=5, batch_size=32)
#y_deep = model_deep.predict(x_test)
#fpr_deep, tpr_deep, ts_deep = roc_curve(y_test, y_deep)
#
#model_deep.fit(x_train_pca, y_train, epochs=5, batch_size=32)
#y_deep_pca = model_deep.predict(x_test_pca)
#fpr_deep_pca, tpr_deep_pca, ts_deep_pca = roc_curve(y_test, y_deep_pca)
#
## display results
#plt.plot(1.0 - fpr_deep, tpr_deep, label="Deep NN")
#plt.plot(1.0 - fpr_deep_pca, tpr_deep_pca, label="Deep NN, PCA")
#plt.legend()
#plt.show()


model_shallow = Sequential()
model_shallow.add(Dense(input_dim = 11, units = 64))
model_shallow.add(Activation('sigmoid'))
model_shallow.add(Dense(units = 1))
model_shallow.add(Activation('sigmoid'))

model_shallow.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_shallow.fit(x_train, y_train, epochs=5, batch_size=32)
y_pred_shallow = model_shallow.predict(x_test)
fpr_shallow, tpr_shallow, ts_shallow = roc_curve(y_test, y_pred_shallow)

model_shallow.fit(x_train_pca, y_train, epochs=5, batch_size=32)
y_pred_shallow_pca = model_shallow.predict(x_test_pca)
fpr_shallow_pca, tpr_shallow_pca, ts_shallow_pca = roc_curve(y_test, y_pred_shallow_pca)

plt.plot(1.0 - fpr_shallow, tpr_shallow, label = "Shallow NN")
plt.plot(1.0 - fpr_shallow_pca, tpr_shallow_pca, label = "Shallow NN")

plt.title("ROC Curve")
plt.xlabel("True Positive Rate")
plt.ylabel("True Negative Rate")
plt.show()
