# Importing libs

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# UNET Class

class Unet:
    def _init_(self, dims):
        self.dims = dims

    def build(self):
        inp = Input(self.dims)

        conv_1 = Conv2D(filters=64, kernel_size=3,
                        padding='same', activation='relu')(inp)
        conv_1 = Conv2D(filters=64, kernel_size=3,
                        padding='same', activation='relu')(conv_1)
        max_pool1 = MaxPool2D(pool_size=(2, 2))(conv_1)

        conv_2 = Conv2D(filters=128, kernel_size=3,
                        padding='same', activation='relu')(max_pool1)
        conv_2 = Conv2D(filters=128, kernel_size=3,
                        padding='same', activation='relu')(conv_2)
        bn_2 = BatchNormalization(axis=3)(conv_2)
        max_pool2 = MaxPool2D(pool_size=(2, 2))(bn_2)

        conv_3 = Conv2D(filters=256, kernel_size=3,
                        padding='same', activation='relu')(max_pool2)
        conv_3 = Conv2D(filters=256, kernel_size=3,
                        padding='same', activation='relu')(conv_3)
        max_pool3 = MaxPool2D(pool_size=(2, 2))(conv_3)

        conv_4 = Conv2D(filters=512, kernel_size=3,
                        padding='same', activation='relu')(max_pool3)
        conv_4 = Conv2D(filters=512, kernel_size=3,
                        padding='same', activation='relu')(conv_4)
        bn_4 = BatchNormalization(axis=3)(conv_4)
        max_pool4 = MaxPool2D(pool_size=(2, 2))(bn_4)

        conv_5 = Conv2D(filters=1024, kernel_size=3,
                        padding='same', activation='relu')(max_pool4)
        conv_5 = Conv2D(filters=1024, kernel_size=3,
                        padding='same', activation='relu')(conv_5)
        drop_5 = Dropout(0.3)(conv_5)

        up_6 = Conv2DTranspose(filters=512, kernel_size=3,
                               strides=(2, 2), padding="same")(drop_5)
        concat_6 = concatenate([up_6, bn_4], axis=3)
        conv_6 = Conv2D(filters=512, kernel_size=3,
                        padding='same', activation='relu')(concat_6)
        conv_6 = Conv2D(filters=512, kernel_size=3,
                        padding='same', activation='relu')(conv_6)
        bn_6 = BatchNormalization(axis=3)(conv_6)

        up_7 = Conv2DTranspose(filters=256, kernel_size=3,
                               strides=(2, 2), padding="same")(bn_6)
        concat_7 = concatenate([up_7, conv_3], axis=3)
        conv_7 = Conv2D(filters=256, kernel_size=3,
                        padding='same', activation='relu')(concat_7)
        conv_7 = Conv2D(filters=256, kernel_size=3,
                        padding='same', activation='relu')(conv_7)

        up_8 = Conv2DTranspose(filters=128, kernel_size=3,
                               strides=(2, 2), padding="same")(conv_7)
        concat_8 = concatenate([up_8, bn_2], axis=3)
        conv_8 = Conv2D(filters=128, kernel_size=3,
                        padding='same', activation='relu')(concat_8)
        conv_8 = Conv2D(filters=128, kernel_size=3,
                        padding='same', activation='relu')(conv_8)
        bn_8 = BatchNormalization(axis=3)(conv_8)

        up_9 = Conv2DTranspose(filters=64, kernel_size=3,
                               strides=(2, 2), padding="same")(bn_8)
        concat_9 = concatenate([up_9, conv_1], axis=3)
        conv_9 = Conv2D(filters=64, kernel_size=3,
                        padding='same', activation='relu')(concat_9)
        conv_9 = Conv2D(filters=64, kernel_size=3,
                        padding='same', activation='relu')(conv_9)

        conv_10 = Conv2D(filters=1, kernel_size=1,
                         activation='sigmoid')(conv_9)

        return Model(inputs=inp, outputs=conv_10)

    def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
        # theta_x(?,g_height,g_width,inter_channel)

        theta_x = Conv2D(inter_channel, [1, 1], strides=[
                         1, 1], data_format=data_format)(x)

        # phi_g(?,g_height,g_width,inter_channel)

        phi_g = Conv2D(inter_channel, [1, 1], strides=[
                       1, 1], data_format=data_format)(g)

        # f(?,g_height,g_width,inter_channel)

        f = Activation('relu')(add([theta_x, phi_g]))

        # psi_f(?,g_height,g_width,1)

        psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

        rate = Activation('sigmoid')(psi_f)

        # rate(?,x_height,x_width)

        # att_x(?,x_height,x_width,x_channel)

        att_x = multiply([x, rate])

        return att_x


class AttnUnet(Unet):

    def attention_up_and_concate(down_layer, layer, features, data_format='channels_last'):
        if data_format == 'channels_last':
            in_channel = down_layer.get_shape().as_list()[1]
        else:
            in_channel = down_layer.get_shape().as_list()[3]

    #     up = Conv2DTranspose(features, [2, 2], strides=[2, 2])(down_layer)
        up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
        layer = attention_block_2d(
            x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

        if data_format == 'channels_last':
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
        else:
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

        concate = my_concat([up, layer])
        return concate

    def att_unet(img_w=240, img_h=240, data_format='channels_last'):
        inputs = Input((240, 240, 2))
        x = inputs
        depth = 4
        features = 32
        skips = []
        for i in range(depth):
            x = Conv2D(features, (3, 3), activation='relu',
                       padding='same', data_format=data_format)(x)
            x = Dropout(0.2)(x)
            x = Conv2D(features, (3, 3), activation='relu',
                       padding='same', data_format=data_format)(x)
            skips.append(x)
            x = MaxPooling2D((2, 2), data_format='channels_last')(x)
            features = features * 2

        x = Conv2D(features, (3, 3), activation='relu',
                   padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu',
                   padding='same', data_format=data_format)(x)
    # x - 512 si - 256
        for i in reversed(range(depth)):
            features = features / 2
            x = attention_up_and_concate(
                x, skips[i], features, data_format=data_format)
            x = Conv2D(features, (3, 3), activation='relu',
                       padding='same', data_format=data_format)(x)
            x = Dropout(0.2)(x)
            x = Conv2D(features, (3, 3), activation='relu',
                       padding='same', data_format=data_format)(x)

        conv6 = Conv2D(4, (1, 1), padding='same',
                       data_format=data_format, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=conv6)

        return model


class ResUnet:

    def res_block(x, nb_filters, strides):
        res_path = BatchNormalization()(x)
        res_path = Activation(activation='relu')(res_path)
        res_path = Conv2D(filters=nb_filters[0], kernel_size=(
            3, 3), padding='same', strides=strides[0])(res_path)
        res_path = BatchNormalization()(res_path)
        res_path = Activation(activation='relu')(res_path)
        res_path = Conv2D(filters=nb_filters[1], kernel_size=(
            3, 3), padding='same', strides=strides[1])(res_path)

        shortcut = Conv2D(nb_filters[1], kernel_size=(
            1, 1), strides=strides[0])(x)
        shortcut = BatchNormalization()(shortcut)

        res_path = add([shortcut, res_path])
        return res_path

    def encoder(x):
        to_decoder = []

        main_path = Conv2D(filters=64, kernel_size=(
            3, 3), padding='same', strides=(1, 1))(x)
        main_path = BatchNormalization()(main_path)
        main_path = Activation(activation='relu')(main_path)

        main_path = Conv2D(filters=64, kernel_size=(
            3, 3), padding='same', strides=(1, 1))(main_path)

        shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(x)
        shortcut = BatchNormalization()(shortcut)

        main_path = add([shortcut, main_path])
        # first branching to decoder
        to_decoder.append(main_path)

        main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
        to_decoder.append(main_path)

        main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
        to_decoder.append(main_path)

        return to_decoder

    def decoder(x, from_encoder):
        main_path = UpSampling2D(size=(2, 2))(x)
        main_path = concatenate([main_path, from_encoder[2]], axis=3)
        main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

        main_path = UpSampling2D(size=(2, 2))(main_path)
        main_path = concatenate([main_path, from_encoder[1]], axis=3)
        main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

        main_path = UpSampling2D(size=(2, 2))(main_path)
        main_path = concatenate([main_path, from_encoder[0]], axis=3)
        main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

        return main_path

    def build_res_unet(input_shape):
        inputs = Input(shape=input_shape)

        to_decoder = encoder(inputs)

        path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

        path = decoder(path, from_encoder=to_decoder)

        path = Conv2D(filters=1, kernel_size=(
            1, 1), activation='sigmoid')(path)

        return Model(input=inputs, output=path)
