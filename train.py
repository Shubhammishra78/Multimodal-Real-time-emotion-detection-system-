
import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 50

def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    def to_image(pixel_str):
        arr = np.fromstring(pixel_str, dtype=int, sep=' ')
        return arr.reshape((48,48))
    X_train, y_train, X_val, y_val = [],[],[],[]
    for _, row in df.iterrows():
        img = to_image(row['pixels'])
        usage = row['Usage']
        emotion = int(row['emotion'])
        if usage == 'Training':
            X_train.append(img)
            y_train.append(emotion)
        else:
            X_val.append(img)
            y_val.append(emotion)
    X_train = np.array(X_train).reshape(-1,48,48,1).astype('float32')/255.0
    X_val = np.array(X_val).reshape(-1,48,48,1).astype('float32')/255.0
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val = to_categorical(y_val, NUM_CLASSES)
    return X_train, y_train, X_val, y_val

def make_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='path to FER2013.csv')
    parser.add_argument('--out', type=str, default='../models/emotion_model.h5')
    args = parser.parse_args()

    print('Loading data from', args.csv)
    X_train, y_train, X_val, y_val = load_fer2013(args.csv)
    print('Data shapes:', X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    model = make_model()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    checkpoint = ModelCheckpoint(args.out, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              steps_per_epoch=max(1, len(X_train)//BATCH_SIZE),
              epochs=EPOCHS,
              validation_data=(X_val, y_val),
              callbacks=[checkpoint, es, rl])

    print('Training finished. Best model saved to', args.out)
