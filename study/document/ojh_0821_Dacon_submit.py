import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# seed
import os
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

train = np.load('data/train.npy', allow_pickle = 'True')
test = np.load('data/test.npy', allow_pickle = 'True')

x = train[:,2:]
x = np.reshape(x, (-1, 28, 28, 1))

x_letter = train[:,1]
x_letter = np.reshape(x_letter, (-1, 1))
en = OneHotEncoder()
x_letter = en.fit_transform(x_letter).toarray()

y = train[:,0]
y = np.reshape(y, (-1, 1))
en = OneHotEncoder()
y = en.fit_transform(y).toarray()

print(x.shape)
print(x_letter.shape)
print(y.shape)

image_generator = ImageDataGenerator(width_shift_range=0.1,
                                     height_shift_range=0.1, 
                                     zoom_range=[0.8,1.2],
                                     shear_range=10)

x_total = x.copy()
def augment(x):
    aug_list = []
    for i in range(x.shape[0]):
        num_aug = 0
        tmp = x[i]
        tmp = tmp.reshape((1,) + tmp.shape)
        for x_aug in image_generator.flow(tmp, batch_size = 1) :
            if num_aug >= 1:
                break
            aug_list.append(x_aug[0])
            num_aug += 1
    aug_list = np.array(aug_list)
    return aug_list

n = 2
for i in range(n):
    arr = augment(x)
    x_total = np.concatenate((x_total, arr), axis=0)
    if i > n:
        break

print(x_total.shape)

y_total = y.copy()
for i in range(n):
    arr = y.copy()
    y_total = np.concatenate((y_total, arr), axis=0)

print(y_total.shape)

x_letter_total = x_letter.copy()
for i in range(n):
    arr = x_letter.copy()
    x_letter_total = np.concatenate((x_letter_total, arr), axis=0)
    
print(x_letter_total.shape)

x_train, x_val, y_train, y_val = train_test_split(x_total, y_total, test_size=0.2, shuffle=True)#, stratify=y_total)
x_letter_train = x_letter_total[:x_train.shape[0],:]
x_letter_val = x_letter_total[x_train.shape[0]:,:]

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
print(x_letter_train.shape)
print(x_letter_val.shape)

def Conv_block(x, growth_rate, activation='relu'):
    x_l = BatchNormalization()(x)
    x_l = Activation(activation)(x_l)
    x_l = Conv2D(growth_rate*4, (1,1), padding='same', kernel_initializer='he_normal')(x_l)
    
    x_l = BatchNormalization()(x_l)
    x_l = Activation(activation)(x_l)
    x_l = Conv2D(growth_rate, (3,3), padding='same', kernel_initializer='he_normal')(x_l)
    
    x = concatenate([x, x_l])
    return x

def Dense_block(x, layers, growth_rate=32):
    for i in range(layers):
        x = Conv_block(x, growth_rate)
    return x

def Transition_layer(x, compression_factor=0.5, activation='relu'):
    reduced_filters = int(tf.keras.backend.int_shape(x)[-1] * compression_factor)
    
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(reduced_filters, (1,1), padding='same', kernel_initializer='he_normal')(x)
    
    x = AveragePooling2D((2,2), padding='same', strides=2)(x)
    return x

def DenseNet(model_input, classes, densenet_type='DenseNet-121'):
    x = Conv2D(base_growth_rate*2, (5,5), padding='same', strides=1,
               kernel_initializer='he_normal')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((2,2), padding='same', strides=1)(x)
    
    x = Dense_block(x, layers_in_block[densenet_type][0], base_growth_rate)
    x = Transition_layer(x, compression_factor=0.5)
    x = Dense_block(x, layers_in_block[densenet_type][1], base_growth_rate)
    x = Transition_layer(x, compression_factor=0.5)
    x = Dense_block(x, layers_in_block[densenet_type][2], base_growth_rate)
    #x = Transition_layer(x, compression_factor=0.5)
    #x = Dense_block(x, layers_in_block[densenet_type][3], base_growth_rate)
    
    x = GlobalAveragePooling2D()(x)
    
    model_output = Dense(classes, activation='softmax', kernel_initializer='he_normal')(x)
    
    model = Model(model_input, model_output, name=densenet_type)
    
    return model

layers_in_block = {'DenseNet-121':[6, 12, 24, 16],
                   'DenseNet-169':[6, 12, 32, 32],
                   'DenseNet-201':[6, 12, 48, 32],
                   'DenseNet-265':[6, 12, 64, 48],
                   'myDenseNet':[8, 12, 16, 32]}

base_growth_rate = 32

model_input = Input(shape=(28,28,1))
classes = 10

model = DenseNet(model_input, classes, 'DenseNet-121')

model.summary()

model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'categorical_crossentropy')

#es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
cp = ModelCheckpoint('./models/{epoch:02d}-{val_acc:.4f}.h5', monitor='val_loss',
                     save_best_only=True, mode='min')

history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val), 
                    batch_size=64, epochs=100, verbose=1, callbacks = [cp])

x_test = test[:,1:]
x_test = np.reshape(x_test, (-1, 28, 28, 1))
print(x_test.shape)

submission = pd.read_csv('data/submission.csv')
submission['digit'] = np.argmax(model.predict(x_test), axis=1)
submission.to_csv('data/submission_densenet(0821).csv', index=False)