from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import BatchNormalization, Lambda, Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers.merging import Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


def color_net(num_classes):
    # placeholder for input image
    i_image = Input(shape=(224, 224, 3))

    # function for making layers
    def my_layer_maker(n_filters, filter_size, prev_layer, strides=(1,1), activation='relu', max_pool=False, batch_norm=False):
        conv_layer = Convolution2D(filters=n_filters, kernel_size=filter_size, strides=strides, activation=activation, padding='same')(prev_layer)
        if batch_norm:
            conv_layer = BatchNormalization()(conv_layer)
        if max_pool:
            conv_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_layer)
        return conv_layer

    # top branch
    top_conv1 = my_layer_maker(n_filters=48, filter_size=(11, 11), prev_layer=i_image, strides=(4, 4), max_pool=True, batch_norm=True)
    top_top_conv2 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, :24])(top_conv1), max_pool=True, batch_norm=True)
    top_bot_conv2 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, 24:])(top_conv1), max_pool=True, batch_norm=True)
    top_conv3 = my_layer_maker(n_filters=192, filter_size=(3, 3), prev_layer=Concatenate()([top_top_conv2, top_bot_conv2]))
    top_top_conv4 = my_layer_maker(n_filters=96, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, :96])(top_conv3))
    top_bot_conv4 = my_layer_maker(n_filters=96, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, 96:])(top_conv3))
    top_top_conv5 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=top_top_conv4, max_pool=True)
    top_bot_conv5 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=top_bot_conv4, max_pool=True)

    # bottom branch
    bottom_conv1 = my_layer_maker(n_filters=48, filter_size=(11, 11), prev_layer=i_image, strides=(4, 4), max_pool=True, batch_norm=True)
    bottom_top_conv2 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, :24])(bottom_conv1), max_pool=True, batch_norm=True)
    bottom_bot_conv2 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, 24:])(bottom_conv1), max_pool=True, batch_norm=True)
    bottom_conv3 = my_layer_maker(n_filters=192, filter_size=(3, 3), prev_layer=Concatenate()([bottom_top_conv2, bottom_bot_conv2]))
    bottom_top_conv4 = my_layer_maker(n_filters=96, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, :96])(bottom_conv3))
    bottom_bot_conv4 = my_layer_maker(n_filters=96, filter_size=(3, 3), prev_layer=Lambda(lambda x: x[:, :, :, 96:])(bottom_conv3))
    bottom_top_conv5 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=bottom_top_conv4, max_pool=True)
    bottom_bot_conv5 = my_layer_maker(n_filters=64, filter_size=(3, 3), prev_layer=bottom_bot_conv4, max_pool=True)

    # connect both branches
    conv_output = Concatenate()([top_top_conv5, top_bot_conv5, bottom_top_conv5, bottom_bot_conv5])
    flatten = Flatten()(conv_output)

    # add fully-connected and last layer
    f1 = Dense(units=4096, activation='relu')(flatten)
    f1 = Dropout(0.6)(f1)
    f2 = Dense(units=4096, activation='relu')(f1)
    f2 = Dropout(0.6)(f2)
    output = Dense(units=num_classes, activation='softmax')(f2)

    model = Model(inputs=i_image, outputs=output)
    sgd = SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


img_rows, img_cols = 224, 224
num_classes = 15
batch_size = 4
nb_epoch = 20

# initialise model
model = color_net(num_classes)

filepath = 'color_weights_test.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_grads=False,
                          write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                          embeddings_data=None, update_freq='batch')
callbacks_list = [checkpoint, tensorboard]

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    dtype='float32')

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  dtype='float32')

training_set = train_datagen.flow_from_directory(
    'Datasets/Color_DataSet/train',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'Datasets/Color_DataSet/test',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')
label_map = (test_set.class_indices)

model.fit(
        training_set,
        steps_per_epoch=len(training_set),
        epochs=nb_epoch,
        validation_data=test_set,
        validation_steps=len(test_set),
        callbacks=callbacks_list)


model.save('color_model.h5')
