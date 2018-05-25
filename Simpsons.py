import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from SimpleDatasetLoader import SimpleDatasetLoader
import glob
import sklearn
import cv2
import os
import scipy
import matplotlib.pyplot as plt
import sklearn.metrics
from mpl_toolkits.axes_grid1 import AxesGrid


map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}
pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)

def create_model_four_conv(input_shape):
    """
    CNN Keras model with 4 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def training(model, X_train, X_test, y_train, y_test):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(X_train)



        filepath="my_simpsons.model4.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        H = model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=30,
                                    callbacks=callbacks_list,
                                    validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=0)
        print('\nKeras CNN #2B - accuracy:', score[1])
        return model , H

def load_model_from_checkpoint(weights_path, input_shape=(pic_size,pic_size,3)):
    model, opt = create_model_four_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model
def load_train_set(dirname):
   X_train = []
   Y_train = []
   for label,character in map_characters.items():
       list_images = os.listdir(dirname+'/'+character)
       for image_name in glob.glob(dirname+'/'+character+'/'+'*.*'):
           image = cv2.imread(image_name)
           X_train.append(cv2.resize(image,(64,64)).astype('float32')/255)
           Y_train.append(label)

   return np.array(X_train), keras.utils.to_categorical(np.array(Y_train), num_classes)
def load_test_set(path):
    pics, labels = [], []
    reverse_dict = {v:k for k,v in map_characters.items()}
    for pic in glob.glob(path+'*.*'):
        char_name = "_".join(pic.split('/')[1].split('_')[:-1])
        if char_name in reverse_dict:
            temp = cv2.imread(pic)
            temp = cv2.resize(temp, (pic_size,pic_size)).astype('float32') / 255.
            pics.append(temp)
            labels.append(reverse_dict[char_name])
    X_test = np.array(pics)
    y_test = np.array(labels)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Test set", X_test.shape, y_test.shape)
    return X_test, y_test

if __name__ == '__main__':
    model = load_model_from_checkpoint('./my_simpsons.model4.hdf5')
    #model = load_model_from_checkpoint('./weights.best.hdf5')

    #train

    X_train_Ori, y_train_Ori = load_train_set("simpsons_dataset")
    (X_train, X_test, y_train, y_test) = train_test_split(X_train_Ori, y_train_Ori,
    	test_size=0.2)
    (model,H)=training(model, X_train, X_test, y_train, y_test)



    #history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 30), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 30), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 30), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 30), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.savefig("loss-accuracy2.png")
    plt.legend()

    #predict
    X_out, y_out = load_test_set("kaggle_simpson_testset/")
    y_pred = model.predict(X_out)


    plt.style.use('classic')

    F = plt.figure( figsize=(15,20))
    grid = AxesGrid(F, 111,
                    nrows_ncols=(4, 5),
                    axes_pad=0,
                    label_mode="1")

    for i in range(18):
        char = map_characters[i]
        image = cv2.imread(np.random.choice(glob.glob('./kaggle_simpson_testset/'+char+'*.*')))
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(image, (64, 64)).astype('float32') / 255.
        a = model.predict(pic.reshape(1, 64, 64,3))[0]
        actual = char.split('_')[0].title()
        text = sorted(['{:s} : {:.1f}%'.format(map_characters[k].split('_')[0].title(), 100*v) for k,v in enumerate(a)],
           key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)[:3]
        img = cv2.resize(img, (352, 352))
        cv2.rectangle(img, (0,260),(215,352),(255,255,255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Actual : %s' % actual, (10, 280), font, 0.7,(0,0,0),2,cv2.LINE_AA)
        for k, t in enumerate(text):
            cv2.putText(img, t,(10, 300+k*18), font, 0.65,(0,0,0),2,cv2.LINE_AA)
        grid[i].imshow(img)
    plt.savefig("char_iden.png")
    #plt.show()
    print('\n', sklearn.metrics.classification_report(np.where(y_out > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')
