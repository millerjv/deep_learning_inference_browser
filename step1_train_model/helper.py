""" For helper code """

import numpy as np

from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, BatchNormalization, Activation
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D, relu6
from keras.models import Model
from keras.optimizers import Adam

from keras import backend as K
K.set_image_dim_ordering('tf')

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def dice_coef(y_true, y_pred, smooth = 1. ):
    # version that returns a single dice coeff for the entire batch
    y_true_f = K.flatten(y_true[:,:,:,0])   # just use foreground
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # version to return a vector of dice coefs, 1 per sample in batch
    #intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    #coef = (2. * intersection + smooth) / (K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) + smooth)
    return coef

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))
    #return 1.0 - dice_coef(y_true, y_pred)

def dice_coef_by_slice(y_true, y_pred, smooth = 1. ):
    intersection = K.sum(y_true * y_pred, axis=[2,3,4])
    coef = (2. * intersection + smooth) / (K.sum(y_true, axis=[2,3,4]) + K.sum(y_pred, axis=[2,3,4]) + smooth)
    coef = K.mean(coef, axis=1)
    return coef

def dice_coef_by_slice_loss(y_true, y_pred):
    return -K.log(dice_coef_by_slice(y_true, y_pred))

def model5_MultiLayer(weights=False,
    filepath="",
    img_rows = 224,
    img_cols = 224,
    n_cl_in=3,
    n_cl_out=3,
    dropout=0.2,
    learning_rate = 0.001,
    print_summary = False):
    """ difference from model: img_rows and cols, order of axis, and concat_axis"""

    inputs = Input((img_rows, img_cols,n_cl_in))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Dropout(dropout)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Dropout(dropout)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_cl_out, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)


    model.compile(optimizer=Adam(lr=learning_rate),
        loss=dice_coef_loss,
        metrics=['accuracy'])

    if weights and len(filepath)>0:
        model.load_weights(filepath)

    if print_summary:
        print (model.summary())

    return model

def model5_mobile_unet(weights=False,
    filepath="",
    img_rows = 224,
    img_cols = 224,
    n_cl_in=3,
    n_cl_out=3,
    dropout=0.2,
    learning_rate = 0.001,
    batch_normalization=True,
    batch_normalization_after=True,
    balanced=True,
    print_summary = False):
    """ difference from model: img_rows and cols, order of axis, concat_axis, batch_normalization and balanced.
        balanced - when merging the down path into the up path, ensure the number of channels from down an up are the same.
    """

    def _conv2D_block(inputs, filters, batch_normalization=True, batch_normalization_after=True, activation='relu', prefix='Conv2DBlock'):
        prefix1 = "{}_{}".format(prefix, 'Conv2DBlock')
        prefix2 = "{}_{}".format(prefix1, K.get_uid(prefix1))

        prefix3 = "{}_{}".format(prefix2, 'Conv2D')
        fullname = "{}_{}".format(prefix3, K.get_uid(prefix3))

        conv = Conv2D(filters, (3, 3), padding='same', use_bias=not batch_normalization, name=fullname)(inputs)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and not batch_normalization_after else conv

        prefix4 = "{}_{}".format(prefix2, 'Activation')
        fullname = "{}_{}".format(prefix4, K.get_uid(prefix4))
        conv = Activation(activation, name=fullname)(conv)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and batch_normalization_after else conv
        return conv

    def _depthwise_conv2d_block(inputs, filters, batch_normalization=True, batch_normalization_after=True, activation='relu', prefix='DepthwiseBlock'):
        prefix1 = "{}_{}".format(prefix, 'DepthwiseConv2DBlock')
        prefix2 = "{}_{}".format(prefix1, K.get_uid(prefix1))

        prefix3 = "{}_{}".format(prefix2, 'DepthwiseConv2D')
        fullname = "{}_{}".format(prefix3, K.get_uid(prefix3))

        conv = DepthwiseConv2D((3, 3), padding='same', use_bias=not batch_normalization, name=fullname)(inputs)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and not batch_normalization_after else conv

        prefix4 = "{}_{}".format(prefix2, 'Activation')
        fullname = "{}_{}".format(prefix4, K.get_uid(prefix4))
        conv = Activation(activation, name=fullname)(conv)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and batch_normalization_after else conv

        prefix3 = "{}_{}".format(prefix2, 'Conv2D')
        fullname = "{}_{}".format(prefix3, K.get_uid(prefix3))
        conv = Conv2D(filters, (1, 1), padding='same', use_bias=not batch_normalization, name=fullname)(conv)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and not batch_normalization_after else conv

        prefix4 = "{}_{}".format(prefix2, 'Activation')
        fullname = "{}_{}".format(prefix4, K.get_uid(prefix4))
        conv = Activation(activation, name=fullname)(conv)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and batch_normalization_after else conv
        return conv

    def _thickness_block(inputs, filters, batch_normalization=True, batch_normalization_after=True, activation='relu', prefix='ThicknessBlock'):
        '''Reduce (or enlarge) the number of features'''
        prefix1 = "{}_{}".format(prefix, 'ThicknessBlock')
        prefix2 = "{}_{}".format(prefix1, K.get_uid(prefix1))

        prefix3 = "{}_{}".format(prefix2, 'Conv2D')
        fullname = "{}_{}".format(prefix3, K.get_uid(prefix3))

        conv = Conv2D(filters, (1, 1), padding='same', use_bias=not batch_normalization, name=fullname)(inputs)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and not batch_normalization_after else conv

        prefix4 = "{}_{}".format(prefix2, 'Activation')
        fullname = "{}_{}".format(prefix4, K.get_uid(prefix4))
        conv = Activation(activation, name=fullname)(conv)
        conv = BatchNormalization(axis=3)(conv) if batch_normalization and batch_normalization_after else conv
        return conv


    print('-'*30)
    print('Mobile U-net: in={}, out={}, balanced={}, batch_normalization={}, batch_normalization_after={}'.format(n_cl_in, n_cl_out, balanced,batch_normalization, batch_normalization_after and batch_normalization))
    print('-'*30)

    # inputs
    inputs = Input((img_rows, img_cols,n_cl_in))

    # down
    conv1 = _conv2D_block(inputs, 32, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='LeftBlock')              # n_cl_in -> 32 channels
    conv1 = _conv2D_block(conv1, 32, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='LeftBlock')               # 32 -> 32 channels

    pool1 = MaxPooling2D(pool_size=(2, 2), name="DownBlock_2_MaxPooling2D_1")(conv1)
    conv2 = _depthwise_conv2d_block(pool1, 64, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='DownBlock_1')     # 32 -> 64 channels
    conv2 = _depthwise_conv2d_block(conv2, 64, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='DownBlock_1')     # 64 -> 64 channels

    pool2 = MaxPooling2D(pool_size=(2, 2), name="DownBlock_3_MaxPooling2D_1")(conv2)
    conv3 = _depthwise_conv2d_block(pool2, 128, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='DownBlock_2')    # 64 -> 128 channels
    conv3 = _depthwise_conv2d_block(conv3, 128, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='DownBlock_2')    # 128 -> 128 channels

    pool3 = MaxPooling2D(pool_size=(2, 2), name="DownBlock_4_MaxPooling2D_1")(conv3)
    conv4 = _depthwise_conv2d_block(pool3, 256, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='DownBlock_3')    # 128 -> 256 channels
    conv4 = _depthwise_conv2d_block(conv4, 256, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='DownBlock_3')    # 256 -> 256 channels

    # bottom
    pool4 = MaxPooling2D(pool_size=(2, 2), name="BottomBlock_MaxPooling2D_1")(conv4)
    conv5 = _depthwise_conv2d_block(pool4, 512, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='BottomBlock')    # 256 -> 512 channels
    conv5 = _depthwise_conv2d_block(conv5, 512, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='BottomBlock')    # 512 -> 512 channels

    # up path
    conv6 = _thickness_block(conv5, 256, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix="UpBlock_1") if balanced else conv5   # 512 -> 256 channels (match the down path)
    up6 = UpSampling2D(size=(2, 2), name="UpBlock_1_UpSampling2D_1")(conv6)
    cat6 = Concatenate(axis=3, name="UpBlock_1_Concatenate_1")([up6, conv4])                                                # 256 + 256 -> 512 channels (balanced), 512 + 256 -> 768 (unbalanced)
    conv6 = _depthwise_conv2d_block(cat6, 256, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_1')     # 512 -> 256 channels (balanced), 768 -> 256 (unbalanced)
    conv6 = _depthwise_conv2d_block(conv6, 256, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_1')    # 256 -> 256 channels

    conv7 = _thickness_block(conv6, 128, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_2') if balanced else conv6   # 256 -> 128 channels (match the down path)
    up7 = UpSampling2D(size=(2, 2), name="UpBlock_2_UpSampling2D_1")(conv7)
    cat7 = Concatenate(axis=3, name="UpBlock_2_Concatenate_1")([up7, conv3])                                                # 128 + 128 -> 256 channels (balanced), 256 + 128 -> 384 (unbalanced)
    conv7 = _depthwise_conv2d_block(cat7, 128, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_2')     # 256 -> 128 channels (balanced), 384 -> 128 (unbalanced)
    conv7 = _depthwise_conv2d_block(conv7, 128, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_2')    # 128 -> 128 channels

    conv8 = _thickness_block(conv7, 64, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_3') if balanced else conv7   # 128 -> 64 channels (match the down path)
    up8 = UpSampling2D(size=(2, 2), name="UpBlock_3_UpSampling2D_1")(conv8)
    cat8 = Concatenate(axis=3, name="UpBlock_3_Concatenate_1")([up8, conv2])                                                # 64 + 64 -> 128 channels (balanced), 128 + 64 -> 192 (unbalanced)
    conv8 = _depthwise_conv2d_block(cat8, 64, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_3')      # 128 -> 64 channels (balanced), 192 -> 64 (unbalanced)
    conv8 = Dropout(dropout, name="UpBlock_3_Dropout_1")(conv8)
    conv8 = _depthwise_conv2d_block(conv8, 64, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_3')     # 64 -> 64 channels

    conv9 = _thickness_block(conv8, 32, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_4') if balanced else conv8     # 64 -> 32 channels (match the down path)
    up9 = UpSampling2D(size=(2, 2), name="UpBlock_4_UpSampling2D_1")(conv9)
    cat9 = Concatenate(axis=3, name="UpBlock_4_Concatenate_1")([up9, conv1])                                                # 32 + 32 -> 64 channels (balanced), 64 + 32 -> 96 (unbalanced)
    conv9 = _depthwise_conv2d_block(cat9, 32, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_4')      # 64 -> 32 channels (balanced), 96 -> 32 (unbalanced)
    conv9 = Dropout(dropout, name="UpBlock_4_Dropout_1")(conv9)
    conv9 = _depthwise_conv2d_block(conv9, 32, batch_normalization=batch_normalization, batch_normalization_after=batch_normalization_after, prefix='UpBlock_4')     # 32 -> 32 channels

    # last
    conv10 = Conv2D(n_cl_out, (1, 1), activation='sigmoid')(conv9) if n_cl_out == 1 else Conv2D(n_cl_out, (1, 1), activation='softmax', name='RightBlock_Conv2D_1')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    print('Last layer activation {}'.format(model.layers[-1].activation))


    model.compile(optimizer=Adam(lr=learning_rate),
        loss=dice_coef_loss,
        metrics=['accuracy', dice_coef])

    if weights and len(filepath)>0:
        model.load_weights(filepath)

    if print_summary:
        print (model.summary())

    return model
