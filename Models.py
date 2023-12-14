# Importing libs

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Dropout, Conv2DTranspose, concatenate, Activation, add, multiply
from tensorflow.keras.models import Model


class UNet:
    def __init__(self, input_dimensions):
        self.input_dimensions = input_dimensions

    def build(self):
        input_tensor = Input(self.input_dimensions)

        # Encoding path
        conv_block1 = self.convolution_block(input_tensor, 64)
        pool1 = MaxPool2D(pool_size=(2, 2))(conv_block1)

        conv_block2 = self.convolution_block(pool1, 128, apply_batchnorm=True)
        pool2 = MaxPool2D(pool_size=(2, 2))(conv_block2)

        conv_block3 = self.convolution_block(pool2, 256)
        pool3 = MaxPool2D(pool_size=(2, 2))(conv_block3)

        conv_block4 = self.convolution_block(pool3, 512, apply_batchnorm=True)
        pool4 = MaxPool2D(pool_size=(2, 2))(conv_block4)

        # Bridge
        bridge = self.convolution_block(pool4, 1024)  # block5
        drop_bridge = Dropout(0.3)(bridge)

        # Decoding path
        up_block6 = self.upconvolution_block(drop_bridge, conv_block4, 512)
        up_block7 = self.upconvolution_block(up_block6, conv_block3, 256)
        up_block8 = self.upconvolution_block(
            up_block7, conv_block2, 128, apply_batchnorm=True)
        up_block9 = self.upconvolution_block(up_block8, conv_block1, 64)

        # Output layer
        output_tensor = Conv2D(filters=1, kernel_size=1,
                               activation='sigmoid')(up_block9)

        return Model(inputs=input_tensor, outputs=output_tensor)

    def convolution_block(self, input_layer, num_filters, apply_batchnorm=False):
        conv_layer = Conv2D(num_filters, (3, 3), padding='same',
                            activation='relu')(input_layer)
        conv_layer = Conv2D(num_filters, (3, 3),
                            padding='same', activation='relu')(conv_layer)
        if apply_batchnorm:
            conv_layer = BatchNormalization(axis=3)(conv_layer)
        return conv_layer

    def upconvolution_block(self, input_layer, concat_layer, num_filters, apply_batchnorm=False):
        up_conv = Conv2DTranspose(num_filters, (3, 3), strides=(
            2, 2), padding='same')(input_layer)
        merged_layer = concatenate([up_conv, concat_layer], axis=3)
        conv_block = self.convolution_block(
            merged_layer, num_filters, apply_batchnorm)
        return conv_block

    @staticmethod
    def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
        theta_x = Conv2D(inter_channel, (1, 1), strides=(
            1, 1), data_format=data_format)(x)
        phi_g = Conv2D(inter_channel, (1, 1), strides=(
            1, 1), data_format=data_format)(g)
        f = Activation('relu')(add([theta_x, phi_g]))
        psi_f = Conv2D(1, (1, 1), strides=(1, 1), data_format=data_format)(f)
        rate = Activation('sigmoid')(psi_f)
        att_x = multiply([x, rate])
        return att_x


class AttnUNet(UNet):

    @staticmethod
    def attention_up_and_concate(down_layer, layer, features, data_format='channels_last'):
        in_channel = down_layer.get_shape().as_list(
        )[3] if data_format == 'channels_last' else down_layer.get_shape().as_list()[1]

        up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
        attention_layer = UNet.attention_block_2d(
            x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

        concat_axis = 3 if data_format == 'channels_last' else 1
        my_concat = Lambda(lambda x: K.concatenate(
            [x[0], x[1]], axis=concat_axis))
        concate = my_concat([up, attention_layer])
        return concate

    def build_attention_unet(self, img_w=240, img_h=240, data_format='channels_last'):
        inputs = Input((img_w, img_h, 2))
        x = inputs
        depth = 4
        features = 32
        skips = []
        for i in range(depth):
            x = self.convolution_block(x, features, data_format=data_format)
            skips.append(x)
            x = MaxPooling2D((2, 2), data_format=data_format)(x)
            features *= 2

        x = self.convolution_block(x, features, data_format=data_format)

        for i in reversed(range(depth)):
            features //= 2
            x = self.attention_up_and_concate(
                x, skips[i], features, data_format=data_format)
            x = self.convolution_block(x, features, data_format=data_format)

        conv6 = Conv2D(4, (1, 1), padding='same',
                       data_format=data_format, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=conv6)
        return model


class ResUNet:

    @staticmethod
    def residual_block(input_tensor, filter_numbers, stride_values):
        residual_path = BatchNormalization()(input_tensor)
        residual_path = Activation('relu')(residual_path)
        residual_path = Conv2D(
            filter_numbers[0], (3, 3), padding='same', strides=stride_values[0])(residual_path)
        residual_path = BatchNormalization()(residual_path)
        residual_path = Activation('relu')(residual_path)
        residual_path = Conv2D(
            filter_numbers[1], (3, 3), padding='same', strides=stride_values[1])(residual_path)

        shortcut = Conv2D(
            filter_numbers[1], (1, 1), strides=stride_values[0])(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        return add([shortcut, residual_path])

    @staticmethod
    def encoding_path(input_tensor):
        decoder_connections = []

        main_path = Conv2D(64, (3, 3), padding='same',
                           strides=(1, 1))(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = Activation('relu')(main_path)
        main_path = Conv2D(64, (3, 3), padding='same',
                           strides=(1, 1))(main_path)

        shortcut = Conv2D(64, (1, 1), strides=(1, 1))(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        main_path = add([shortcut, main_path])
        decoder_connections.append(main_path)

        main_path = ResUNet.residual_block(
            main_path, [128, 128], [(2, 2), (1, 1)])
        decoder_connections.append(main_path)

        main_path = ResUNet.residual_block(
            main_path, [256, 256], [(2, 2), (1, 1)])
        decoder_connections.append(main_path)

        return decoder_connections

    @staticmethod
    def decoding_path(input_tensor, encoder_outputs):
        main_path = UpSampling2D(size=(2, 2))(input_tensor)
        main_path = concatenate([main_path, encoder_outputs[2]], axis=3)
        main_path = ResUNet.residual_block(
            main_path, [256, 256], [(1, 1), (1, 1)])

        main_path = UpSampling2D(size=(2, 2))(main_path)
        main_path = concatenate([main_path, encoder_outputs[1]], axis=3)
        main_path = ResUNet.residual_block(
            main_path, [128, 128], [(1, 1), (1, 1)])

        main_path = UpSampling2D(size=(2, 2))(main_path)
        main_path = concatenate([main_path, encoder_outputs[0]], axis=3)
        main_path = ResUNet.residual_block(
            main_path, [64, 64], [(1, 1), (1, 1)])

        return main_path

    @staticmethod
    def build_res_unet(input_shape):
        inputs = Input(shape=input_shape)

        encoder_outputs = ResUNet.encoding_path(inputs)

        path = ResUNet.residual_block(
            encoder_outputs[2], [512, 512], [(2, 2), (1, 1)])
        path = ResUNet.decoding_path(path, encoder_outputs)

        path = Conv2D(1, (1, 1), activation='sigmoid')(path)

        return Model(inputs=inputs, outputs=path)
