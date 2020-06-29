import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


batch_size = 128
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150

model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()

train_datagen = ImageDataGenerator(rescale = 1./255,       # 圖片做標準化
                                   shear_range = 0.2,      # 做圖片偏移
                                   zoom_range = 0.2,       # 放大縮小
                                   horizontal_flip = True) # 水平翻轉

test_datagen = ImageDataGenerator(rescale = 1./255)  #圖片做標準化

training_set = train_datagen.flow_from_directory('F:/python-project/dataset1/train',
                                                 target_size = (64, 64),
                                                 batch_size = 128,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('F:/python-project/dataset1/test',
                                            target_size = (64, 64),
                                            batch_size = 128,
                                            class_mode = 'categorical')

#fit the model
history = classifier.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 10)

#save weights
classifier.save("406410208.h5")