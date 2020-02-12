from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf 
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

def train():
    model=Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3), data_format='channels_first',padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    batch_size=16

    #Augmentation for train and test data to improve accuracy
    train_data=ImageDataGenerator( rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_data=ImageDataGenerator(rescale=1./255)

    train_gen = train_data.flow_from_directory(
        'dataset/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    validation_gen = test_data.flow_from_directory(
        'dataset/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_gen,
        steps_per_epoch=2000// batch_size,
        epochs=60,
        validation_data=validation_gen,
        validation_steps=800 // batch_size)
    model.save_weights('./model/seefood.h5')
    
train()



