from explore         import IciData
from keras.models    import Sequential
from keras.layers    import Dense, Activation
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

training_data = IciData('/home/simonpf/projects/ici/data/sets/full/train.nc')
test_data     = IciData('/home/simonpf/projects/ici/data/sets/full/test.nc')

# Set up the model
model = Sequential()
model.add(Dense(input_dim = 11, units = 32))
model.add(Activation('relu'))
model.add(Dense( units = 32))
model.add(Activation('relu'))
model.add(Dense( units = 32))
model.add(Activation('relu'))
model.add(Dense( units = 32))
model.add(Activation('relu'))
model.add(Dense( units = 32))
model.add(Dense(units = 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(training_data.get_input_data(), training_data.get_output_data("clear_sky"), epochs=5, batch_size=256)

y_pred = model.predict(test_data.get_input_data())
fpr, tpr, ts = roc_curve(test_data.get_output_data('clear_sky'), y_pred)
plt.plot(1.0 - fpr, tpr)

model = Sequential()
model.add(Dense(input_dim = 11, units = 64))
model.add(Activation('linear'))
model.add(Dense(units = 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(training_data.get_input_data(), training_data.get_output_data("clear_sky"), epochs=5, batch_size=256)

y_pred_2 = model.predict(test_data.get_input_data())
fpr_2, tpr_2, ts_2 = roc_curve(test_data.get_output_data('clear_sky'), y_pred_2)
plt.plot(1.0 - fpr_2, tpr_2)
plt.show()
