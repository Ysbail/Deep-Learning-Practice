from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


# initializing
clf = Sequential()

# convolution 
clf.add(Conv2D(32,(3,3), input_shape = (64, 64, 3), activation='relu'))

# pooling
clf.add(MaxPooling2D(pool_size=(2, 2)))

# convolution
clf.add(Conv2D(32,(3,3), activation = 'relu'))

# pooling
clf.add(MaxPooling2D(pool_size = (2, 2)))

# flattening
clf.add(Flatten())

# full connection
clf.add(Dense(128, activation='relu'))

clf.add(Dense(1, activation='sigmoid'))

# compiling
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# steps_per_epoch = samples_per_epoch/ batch_size = 8000/32 = 250
# validation_steps = nb_val_samples / batch_size = 2000/32 = 62.5
clf.fit_generator(train_set,
                  steps_per_epoch=250,
                  epochs=25,
                  validation_data=test_set,
                  validation_steps=62.5)

y_pred = clf.predict(test_set)
y_pred_yes = y_pred > 0.5

