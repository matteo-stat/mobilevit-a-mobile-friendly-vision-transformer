import keras 
import tensorflow as tf
import yaml

with open('config/mobilevit.yml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

shape = (256, 256, 3)

def mobilenetv2_bottleneck_block(
        layer: keras.Layer,
        expansion_factor: int,
        output_channels: int,
        downsampling: bool,
        block_counter: int,
    ) -> keras.Layer:
    """
    mobilenetv2 block slightly modified with swish activation function as described by mobilevit paper

    Args:
        layer (keras.Layer): input layer
        expansion_factor (int): mobilenetv2 expansion factor used to expand channels
        output_channels (int): number of output channels
        downsampling (bool): if True will use stride=2 to perform downsampling, otherwise will use stride=1 and no downsampling will be performed.
        block_counter (int): a progressive counter used for give meaningful names to basic layers in a mobilenetv2 block

    Returns:
        keras.Layer: output from a mobilenetv2 block
    """
    name_prefix = f'block{str(block_counter).zfill(2)}-mv2'

    # save input channels and input layer
    input_channels = layer.shape[-1]
    layer_input = layer

    # expand channels with pointwise convolution, followed by batchnormalization and swish activation function
    layer = keras.layers.Conv2D(filters=input_channels*expansion_factor, kernel_size=1, use_bias=False, name=f'{name_prefix}-expand-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-expand-batchnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-expand-swish')(layer)

    # process with depthwise convolution, followed by batchnormalization and swish activation function
    layer = keras.layers.DepthwiseConv2D(kernel_size=3, strides=2 if downsampling else 1, padding='same', use_bias=False, name=f'{name_prefix}-depth-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-depth-batchnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-depth-swish')(layer)

    # project channels with pointwise convolution, followed by batchnormalization
    layer = keras.layers.Conv2D(filters=output_channels, kernel_size=1, use_bias=False, name=f'{name_prefix}-project-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-project-batchnorm')(layer)

    # optionally add a residual connection
    if not downsampling and input_channels==output_channels:
        layer = keras.layers.Add(name=f'{name_prefix}-residual')([layer, layer_input])

    return layer

# start with conv2d, kernel 3x3, stride=0, poi mv2 e mvit blocks
# swish activation function
# n = 3 nei blocchi mobilevit
# spatial dimension of the feature maps multiple of 2, h,w <= n.. quindi h=2 e w=2 in tutti gli spatial levels
# expansion factor per mobilenetv2 blocks = 4 in tutti, tranne per mobilevit xxs = 2
# transformer layer prende come input d canali, il primo feed forward layer fa un output 2d invece che 4d come solitamente nei transformer


layer_input = keras.Input(shape=shape, dtype=tf.float32, name='model-input')
layer = keras.layers.Rescaling(scale=1./127.5, offset=-1, name='input-rescaling')(layer_input)

# initial convolution block
layer = keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', use_bias=False, name=f'init-conv')(layer)
layer = keras.layers.BatchNormalization(name=f'init-batchnorm')(layer)
layer = keras.layers.Activation(activation='silu', name=f'init-swish')(layer)

# build mobilevit

for block_counter, options in config['xxs']['layers'].items():
    if options['block type'] == 'mobilenetv2':
        layer = mobilenetv2_bottleneck_block(
            layer=layer,
            expansion_factor=options['expansion factor'],
            output_channels=options['output channels'],
            downsampling=options['downsampling'],
            block_counter=block_counter
        )


# mobilevit block
transformer_dimension = 64
name_prefix = f'block{str(block_counter).zfill(2)}-mvit'

# keep a copy of the input layer
layer_input = layer

# local representation
layer = keras.layers.Conv2D(filters=layer.shape[-1], kernel_size=3, padding='same', use_bias=False, name=f'{name_prefix}-locfeat-conv')(layer)
layer = keras.layers.BatchNormalization(name=f'{name_prefix}-locfeat-batchnorm')(layer)
layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-locfeat-swish')(layer)
layer = keras.layers.Conv2D(filters=transformer_dimension, kernel_size=1, use_bias=False, name=f'{name_prefix}-locfeat-conv-expand')(layer)

# unfold
patch_area = int(options['patch width'] * options['patch height'])
number_of_patches = int(layer.shape[1] * layer.shape[2] / patch_area)
layer = keras.layers.Reshape(target_shape=(patch_area, number_of_patches, transformer_dimension), name=f'{name_prefix}-unfold')(layer)

# multi head attention
name_transformer_suffix = '00'
layer = keras.layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}-trf{name_transformer_suffix}-prenorm')(layer)
layer = keras.layers.MultiHeadAttention(
    num_heads=options['number of heads'],
    key_dim=options['transformer dimension'],
    dropout=0.1,
    name=f'{name_prefix}-trf{name_transformer_suffix}-prenorm'
)(layer, layer)

# add and normalization
layer = keras.layers.Add(name=f'{name_prefix}-trf{name_transformer_suffix}-add')([])
layer 

# feed forward


layer_output = layer
model = keras.Model(inputs=layer_input, outputs=layer_output)
model.summary()


