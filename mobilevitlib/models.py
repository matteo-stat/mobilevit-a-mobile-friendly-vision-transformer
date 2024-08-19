import keras
import tensorflow as tf
from typing import Tuple, Optional, Literal
from mobilevitlib import blocks
import yaml

def build_mobilevit(
        network_size: Literal['xxs', 'xs', 's'],
        input_shape: Tuple[int, int, int],
        num_classes: int,        
        rescale_inputs: bool = True,
        use_l2_regularization: bool = True,
    ) -> keras.Model:
    """
    build mobilevit network

    Args:
        network_size (Literal['xxs', 'xs', 's']): network size can be 'xxs', 'xs' or 's'
        input_shape (Tuple[int, int, int]): tuple describing network input shape (height, width, channels)
        num_classes (int): number of classes for the classification head
        rescale_inputs (bool): if True rescale inputs between -1 and 1, otherwise not. Defaults to True.
        use_l2_regularization (bool): if True apply l2 regularizer, otherwise not. Defaults to True.

    Returns:
        keras.Model: a mobilevit network for classification
    """

    # read config
    with open('config/mobilevit.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # input layer
    input_layer = keras.Input(shape=input_shape, dtype=tf.float32, name='model-input')

    # rescaling
    if rescale_inputs:
        input_layer = keras.layers.Rescaling(scale=1./127.5, offset=-1, name='input-rescaling')(input_layer)

    # l2 regularizer
    if use_l2_regularization:
        l2_regularizer = keras.regularizers.L2()
    else:
        l2_regularizer = None

    # initial strided convolution block
    layer = keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_regularizer=l2_regularizer, name=f'init-conv')(input_layer)
    layer = keras.layers.BatchNormalization(name=f'init-batchnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'init-silu')(layer)

    # build mobilevit
    for block_counter, options in config[network_size]['blocks'].items():
        if options['block type'] == 'mobilenetv2':
            layer = blocks.mobilenetv2_bottleneck_block(
                layer=layer,
                expansion_factor=options['expansion factor'],
                output_channels=options['output channels'],
                downsampling=options['downsampling'],
                use_l2_regularization=use_l2_regularization,
                block_counter=block_counter
            )
        elif options['block type'] == 'mobilevit':
            layer = blocks.mobilevit_block(
                layer=layer,
                patch_width=options['patch width'],
                patch_height=options['patch height'],
                transformer_channels=options['transformer channels'],
                transformer_blocks=options['transformer blocks'],
                number_of_heads=options['number of heads'],
                feed_forward_network_units=options['feed forward network units'],
                use_l2_regularization=use_l2_regularization,
                block_counter=block_counter
            )

    # final convolution block
    layer = keras.layers.Conv2D(filters=4*layer.shape[-1], kernel_size=1, use_bias=False, kernel_regularizer=l2_regularizer, name=f'last-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'last-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'last-silu')(layer)

    # classification head
    layer = keras.layers.GlobalAveragePooling2D(name='class-head-global-avg-pool')(layer)
    layer = keras.layers.Dense(units=num_classes, kernel_regularizer=l2_regularizer, name='class-head-dense')(layer)
    layer_output = keras.layers.Softmax(name='class-head-softmax')(layer)

    # model
    model = keras.Model(inputs=input_layer, outputs=layer_output)

    return model
