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
    ) -> keras.Model:
    """
    build mobilevit-v1 network

    Args:
        network_size (Literal[&#39;xxs&#39;, &#39;xs&#39;, &#39;s&#39;]): _description_
        input_shape (Tuple[int, int, int]): _description_
        num_classes (int): _description_
        rescale_inputs (bool, optional): _description_. Defaults to True.

    Returns:
        keras.Model: _description_
    """

    # read config
    with open('config/mobilevit.yml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # input layer
    input_layer = keras.Input(shape=input_shape, dtype=tf.float32, name='model-input')

    # rescaling
    if rescale_inputs:
        input_layer = keras.layers.Rescaling(scale=1./127.5, offset=-1, name='input-rescaling')(input_layer)

    # initial strided convolution block
    layer = keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False, name=f'init-conv')(input_layer)
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
                block_counter=block_counter,
            )

    # final convolution block
    layer = keras.layers.Conv2D(filters=4*layer.shape[-1], kernel_size=1, use_bias=False, name=f'last-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'last-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'last-silu')(layer)

    # classification head
    layer = keras.layers.GlobalAveragePooling2D(name='class-head-global-avg-pool')(layer)
    layer = keras.layers.Dense(units=num_classes, name='class-head-dense')(layer)
    layer_output = keras.layers.Softmax(name='class-head-softmax')(layer)

    # model
    model = keras.Model(inputs=input_layer, outputs=layer_output)

    return model
