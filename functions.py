from tensorflow.keras import layers
import keras
import tensorflow.keras as K
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


"""
    VARIABLES
"""
seed= 157
img_size = (256, 320)
alpha = 0.8
gamma = 2


"""
    PREPARE DATASET
"""
def get_train_test_dataset(df, types='all', test_size=0.1):
    if types != 'all':
        df = df[df['type']==types]
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle = True, random_state =seed, stratify=df['wear'].values)
    
    """
    d_train = (
        np.concatenate( (df_train['image_path'].values,  df_aug_train['image_path'].values), axis = 0), 
        np.concatenate( (df_train['mask_path'].values , df_aug_train['image_path'].values), axis = 0), 
        np.concatenate( (df_train['wear_path'].values , df_aug_train['image_path'].values), axis = 0),
        np.concatenate( (df_train['broken_path'].values , df_aug_train['image_path'].values), axis = 0),
    )
    """
    d_train = (
        df_train['image_path'].values,  
        df_train['mask_path'].values , 
        df_train['wear_path'].values , 
        df_train['broken_path'].values ,
    )
    
    d_test = (
        df_test['image_path'].values , 
        df_test['mask_path'].values, 
        df_test['wear_path'].values , 
        df_test['broken_path'].values ,
    )
    
    data_train = tf.data.Dataset.from_tensor_slices(d_train)
    data_test = tf.data.Dataset.from_tensor_slices(d_test)
    return df_train, df_test, data_train, data_test


"""
    PREPROCESSING
"""
def decode_tool_img(img):
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, (img_size[0], img_size[1]))
    #img = tf.cast(img, tf.float32)
    img = tf.multiply(img, 1. / 255.0)
    return img

def decode_tool_target(img):
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (img_size[0], img_size[1])) #size_0, size_1, 1
    img = tf.reshape(img, [img_size[0], img_size[1]]) #size_0, size_1
    mask_0 = tf.cast(tf.math.less(img, 200), tf.float32 )
    mask = tf.cast(tf.math.greater(img, 200), tf.float32 )
    return mask, mask_0


"""
    POSTPROCESSING
"""
def get_microm(image, magnification, df_scale):
    scale = df_scale[df_scale['magnification']==magnification].iloc[0]
    wear_pixels = np.count_nonzero(image == 1)
    wear_micro_m = (wear_pixels * scale.scale)/scale.pixels
    return wear_micro_m


def get_wear_parameters(mask_input):
    mask_input = (mask_input).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = []
    big_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>big_area:
            biggest_contour = cnt
            big_area = area
    angle=0
    vb= 0
    vb_max = 0
    vb_max_x = 0
    vb_max_y = 0
    vb_base = 0
    rotated = False
    lines_f = [[],[],[]]
    points_f = []
    if big_area>0:
        mask = np.zeros(mask_input.shape,np.uint8)
        cv2.drawContours(mask,[biggest_contour],0,1,-1)
        _, (w_bb, h_bb), angle = cv2.minAreaRect(biggest_contour)   
        if h_bb>w_bb and abs(90-angle)>5:
            rotated = True
            (h, w) = mask.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), 90+angle, 1.0)
            mask = cv2.warpAffine(mask, M, (w, h))
        lines = [[],[],[]]
        points = []
        for i in range(mask.shape[1]):
            section = mask[:,i:i+1]
            rows, cols= np.where(section==1)
            if len(rows)>2:
                lines[0].append(rows[0])
                lines[1].append(rows[-1])
                lines[2].append(rows[-1]-rows[0])
                points.append(i)
        vector = np.array(lines[0])       
        vb_base = np.median(vector)
        for l_1, l_2, l_3, p in zip(lines[0], lines[1], lines[2], points):
            diff = abs(l_1-vb_base)
            if diff<20:
                lines_f[0].append(l_1)
                lines_f[1].append(l_2)
                lines_f[2].append(l_2-l_1)
                points_f.append(p)
        if len(points_f)>2:
            points_dist = [points_f[i+1]-points_f[i] for i in range(len(points_f)-1)]
            points_index = np.where(np.array(points_dist)>10)[0]
            if len(points_index)>1:
                points_index = np.concatenate(([0],points_index))
                points_index = np.concatenate((points_index, [len(points_dist)]))
                points_diff = [points_index[i+1]-points_index[i] for i in range(len(points_index)-1)]
                index = np.where(points_diff == np.max(points_diff))[0]
                begin = points_index[index][0]+1
                end = points_index[index+1][0]+1
                points_f = points_f[begin:end]
                lines_f[0] = lines_f[0][begin:end]
                lines_f[1] = lines_f[1][begin:end]
                lines_f[2] = lines_f[2][begin:end]
            vb_base = sum(lines[0]) / len(lines[0])
            vb = sum(lines_f[2]) / len(lines_f[2])
            vb_max = max(lines_f[2])
            vb_max_index = lines_f[2].index(vb_max)
            vb_max_x = lines_f[1][vb_max_index]
            vb_max_y = points_f[vb_max_index]
        
    return vb, vb_max, vb_max_x, vb_max_y, vb_base, lines_f, points_f, angle, rotated


"""
    GET MODEL
"""
def get_unet_mini(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = layers.concatenate([drop4,up6], axis = 3)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = layers.Conv2D(num_classes, 3, padding="same", activation="sigmoid")(conv9)

    model = keras.Model(input = inputs, output = conv10)


def get_unet_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, padding="same", activation="softmax")(x)
    #layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_unet(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = layers.concatenate([drop4,up6], axis = 3)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = layers.Conv2D(num_classes, 3, padding="same", activation="softmax")(conv9)

    model = keras.Model(input = inputs, output = conv10)

"""
    LOSS FUNCTIONS
"""

def dice_score(y_true, y_pred):
    smooth = K.backend.epsilon()
    y_true_f = K.backend.flatten(y_true)
    y_pred_f = K.backend.flatten(y_pred)
    intersection = K.backend.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.backend.sum(y_true_f) + K.backend.sum(y_pred_f) + smooth)
    return answer

def dice_loss(y_true, y_pred):
    answer = 1. - dice_score(y_true, y_pred)
    return answer

def dice_eval(y_true, y_pred):
    y_true_th = K.backend.cast(K.backend.greater(y_true, 0.5), 'float32')
    y_pred_th = K.backend.cast(K.backend.greater(y_pred, 0.5), 'float32')
    return dice_score(y_true_th, y_pred_th)

def dice_weighted_loss(w_all, w_wear, w_broken):
    def dice_weighted_loss_(y_true, y_pred):
        focal_loss = dice_loss(y_true, y_pred)
        focal_loss_channel2 = dice_loss(y_true[:,:,:,2], y_pred[:,:,:,2])
        focal_loss_channel3 = dice_loss(y_true[:,:,:,3], y_pred[:,:,:,3])
        return w_all * focal_loss + w_wear * focal_loss_channel2 + w_broken * focal_loss_channel3

    return dice_weighted_loss_


def dice_loss_wear(y_true, y_pred):
    answer = 1. - dice_score(y_true[:,:,:2], y_pred[:,:,:2])
    return answer

def binary_crossentropy(y_true, y_pred):
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False))
    return loss

def cross_weighted_loss(w_all, w_wear, w_broken):
    def cross_weighted_loss_(y_true, y_pred):
        focal_loss = binary_crossentropy(y_true, y_pred)
        focal_loss_channel2 = binary_crossentropy(y_true[:,:,:,2], y_pred[:,:,:,2])
        focal_loss_channel3 = binary_crossentropy(y_true[:,:,:,3], y_pred[:,:,:,3])
        return w_all * focal_loss + w_wear * focal_loss_channel2 + w_broken * focal_loss_channel3

    return cross_weighted_loss_

def cross_and_dice_loss(w_cross, w_dice):
    def cross_and_dice_loss_(y_true, y_pred):
        cross_entropy_value = binary_crossentropy(y_true, y_pred)
        dice_loss_value = dice_loss(y_true, y_pred)
        return w_dice * dice_loss_value + w_cross * cross_entropy_value

    return cross_and_dice_loss_

def cross_dice_focal_loss(w_cross, w_dice, w_focal):
    def cross_dice_focal_loss_(y_true, y_pred):
        cross_entropy_value = binary_crossentropy(y_true, y_pred)
        focal_loss_value = focal_loss(y_true, y_pred)
        dice_loss_value = dice_loss(y_true, y_pred)
        return w_focal * focal_loss_value + w_cross * cross_entropy_value + w_dice * dice_loss_value

    return cross_dice_focal_loss_

def cross_and_focal_loss(w_cross, w_focal):
    def cross_and_focal_loss_(y_true, y_pred):
        cross_entropy_value = binary_crossentropy(y_true, y_pred)
        focal_loss_value = focal_loss(y_true, y_pred)
        return w_focal * focal_loss_value + w_cross * cross_entropy_value

    return cross_and_focal_loss_

def focal_and_dice_loss(w_cross, w_dice):
    def cross_and_dice_loss_(y_true, y_pred):
        cross_entropy_value = focal_loss(y_true, y_pred)
        dice_loss_value = dice_loss(y_true, y_pred)
        return w_dice * dice_loss_value + w_cross * cross_entropy_value

    return cross_and_dice_loss_


def focal_and_iou_loss(w_cross, w_dice):
    def focal_and_iou_loss_(y_true, y_pred):
        cross_entropy_value = focal_loss(y_true, y_pred)
        dice_loss_value = iou_loss(y_true, y_pred)
        return w_dice * dice_loss_value + w_cross * cross_entropy_value

    return focal_and_iou_loss_

def focal_weighted_loss(w_all, w_wear, w_broken):
    def focal_weighted_loss_(y_true, y_pred):
        focal_loss = focal_loss(y_true, y_pred)
        focal_loss_channel2 = focal_loss(y_true[:,:,:,2], y_pred[:,:,:,2])
        focal_loss_channel3 = focal_loss(y_true[:,:,:,3], y_pred[:,:,:,3])
        return w_all * focal_loss + w_wear * focal_loss_channel2 + w_broken * focal_loss_channel3

    return focal_weighted_loss_

def focal_loss(targets, inputs):    
    
    inputs = K.backend.flatten(inputs)
    targets = K.backend.flatten(targets)
    
    BCE = K.backend.binary_crossentropy(targets, inputs)
    BCE_EXP = K.backend.exp(-BCE)
    focal_loss = K.backend.mean(alpha * K.backend.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss


def focal_loss_wear(y_true, y_pred):
    smooth = K.backend.epsilon()
    y_true_f = K.backend.flatten(y_true[:,:,:,2])
    y_pred_f = K.backend.flatten(y_pred[:,:,:,2])
    BCE = K.backend.binary_crossentropy(y_true_f, y_pred_f)
    BCE_EXP = K.backend.exp(-BCE)
    focal_loss = K.backend.mean(alpha * K.backend.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss

def iou_metric(y_true, y_pred):
    smooth = K.backend.epsilon()  
    intersection = K.backend.sum(K.backend.abs(y_true * y_pred), axis=[1,2])
    union = K.backend.sum(y_true,[1,2])+K.backend.sum(y_pred,[1,2])-intersection
    IoU = K.backend.mean((intersection + smooth) / (union + smooth), axis=0)
    return IoU
    
def iou_loss(targets, inputs):    
    return 1 - iou_metric(targets, inputs)

def iou_wear_metric(y_true, y_pred):
    return iou_metric(y_true[:,:,:,2], y_pred[:,:,:,2])

def iou_broken_metric(y_true, y_pred):
    return iou_metric(y_true[:,:,:,3], y_pred[:,:,:,3])

def iou_weighted_loss(w_all, w_wear, w_broken):
    def iou_weighted_loss_(y_true, y_pred):
        focal_loss = iou_loss(y_true[:,:,:,1], y_pred[:,:,:,1])
        focal_loss_channel2 = iou_loss(y_true[:,:,:,2], y_pred[:,:,:,2])
        focal_loss_channel3 = iou_loss(y_true[:,:,:,3], y_pred[:,:,:,3])
        return w_all * focal_loss + w_wear * focal_loss_channel2 + w_broken * focal_loss_channel3

    return iou_weighted_loss_

def iou_categorical_loss(w_iou, w_cce):
    def iou_categorical_loss_(y_true, y_pred):
        iou_loss = iou_loss(y_true, y_pred)
        cce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False))
        return w_iou * iou_loss + w_cce * cce_loss 
    return iou_categorical_loss_

def focal_categorical_loss(w_focal, w_cce):
    def focal_categorical_loss_(y_true, y_pred):
        focal_loss = focal_loss(y_true, y_pred)
        cce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False))
        return w_focal * focal_loss + w_cce * cce_loss
    focal_categorical_loss_.__name__ = 'focal_cce'
    return focal_categorical_loss_

def categorical_weighted_loss(w_cce, w_wear, w_broken):
    def categorical_weighted_loss_(y_true, y_pred):
        bce_wear = binary_crossentropy(y_true[:,:,:,2], y_pred[:,:,:,2])
        bce_broken = binary_crossentropy(y_true[:,:,:,3], y_pred[:,:,:,3])
        cce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False))        
        return w_cce * cce_loss + w_wear * bce_wear +  w_broken * bce_broken
    return categorical_weighted_loss_