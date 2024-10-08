import keras

def mobilenetv2_bottleneck_block(
        layer: keras.Layer,
        expansion_factor: int,
        output_channels: int,
        downsampling: bool,
        use_l2_regularization: bool,
        block_counter: int,
    ) -> keras.Layer:
    """
    mobilenetv2 bottleneck block with swish activation function instead of relu6

    Args:
        layer (keras.Layer): input layer
        expansion_factor (int): expansion factor
        output_channels (int): number of output channels
        downsampling (bool): if True will use stride=2 to perform downsampling, otherwise will use stride=1
        use_l2_regularization (bool): if True apply l2 regularizer, otherwise not
        block_counter (int): a progressive counter for assigning meaningful names to basic layers

    Returns:
        keras.Layer: output from a mobilenetv2 block
    """
    # name prefix for basic layers    
    name_prefix = f'block{str(block_counter).zfill(2)}-mv2'

    # save input channels and input layer
    input_channels = layer.shape[-1]
    input_layer = layer

    # l2 regularizer
    if use_l2_regularization:
        l2_regularizer = keras.regularizers.L2()
    else:
        l2_regularizer = None

    # expand channels with pointwise convolution, followed by batch normalization and silu activation function
    layer = keras.layers.Conv2D(filters=input_channels*expansion_factor, kernel_size=1, use_bias=False, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-exp-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-exp-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-exp-silu')(layer)

    # process with depthwise convolution, followed by batch normalization and silu activation function
    layer = keras.layers.DepthwiseConv2D(kernel_size=3, strides=2 if downsampling else 1, padding='same', use_bias=False, depthwise_regularizer=l2_regularizer, name=f'{name_prefix}-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-silu')(layer)

    # project channels with pointwise convolution, followed by batch normalization
    layer = keras.layers.Conv2D(filters=output_channels, kernel_size=1, use_bias=False, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-proj-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-proj-bnorm')(layer)

    # optionally add a residual connection
    if not downsampling and input_channels==output_channels:
        layer = keras.layers.Add(name=f'{name_prefix}-resid')([layer, input_layer])

    return layer

def transformer_block(
        layer: keras.Layer,
        transformer_channels: int,
        number_of_heads: int,
        attention_dropout_rate: float,
        dropout_rate: float,
        feed_forward_network_units: int,
        feed_forward_dropout_rate: float,
        use_l2_regularization: bool,
        name_prefix: str,
    ) -> keras.Layer:
    """
    transformer block

    Args:
        layer (keras.Layer): input layer        
        transformer_channels (int): number of channels to use in the transformer block
        number_of_heads (int): number of heads for the attention layer
        attention_dropout_rate (float): dropout rate for the attention layer
        dropout_rate (float): dropout rate between attention layer and feed forward network
        feed_forward_network_units (int): number of units in the feed forward network
        feed_forward_dropout_rate (float): dropout rate for the feed forward network
        use_l2_regularization (bool): if True apply l2 regularizer, otherwise not
        name_prefix (str): name prefix for assigning meaningful names to basic layers

    Returns:
        keras.Layer: output from transformer block
    """
    # l2 regularizer
    if use_l2_regularization:
        l2_regularizer = keras.regularizers.L2()
    else:
        l2_regularizer = None

    # multi head attention
    input_layer_mha = layer
    layer = keras.layers.LayerNormalization(name=f'{name_prefix}-mha-lnorm')(layer)
    layer = keras.layers.MultiHeadAttention(num_heads=number_of_heads, key_dim=transformer_channels//number_of_heads, dropout=attention_dropout_rate, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-mha')(layer, layer)
    layer = keras.layers.Dropout(rate=dropout_rate, name=f'{name_prefix}-mha-dropout')(layer)
    layer = keras.layers.Add(name=f'{name_prefix}-mha-add')([layer, input_layer_mha])

    # feed forward network
    input_layer_ffn = layer
    layer = keras.layers.LayerNormalization(name=f'{name_prefix}-ffn-lnorm')(layer)
    layer = keras.layers.Dense(units=feed_forward_network_units, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-ffn-exp-dense')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-ffn-exp-silu')(layer)
    layer = keras.layers.Dropout(rate=feed_forward_dropout_rate, name=f'{name_prefix}-ffn-exp-dropout')(layer)
    layer = keras.layers.Dense(units=transformer_channels, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-ffn-proj-dense')(layer)
    layer = keras.layers.Dropout(rate=dropout_rate, name=f'{name_prefix}-ffn-dropout')(layer)
    layer = keras.layers.Add(name=f'{name_prefix}-ffn-add')([layer, input_layer_ffn])

    return layer

def mobilevit_block(
        layer: keras.Layer,
        patch_width: int,
        patch_height: int,
        transformer_channels: int,
        transformer_blocks: int,
        number_of_heads: int,
        feed_forward_network_units: int,
        use_l2_regularization: bool,
        block_counter: int,
        attention_dropout_rate: float = 0.1,
        feed_forward_network_dropout_rate: float = 0.1,
        dropout_rate: float = 0.1,
    ) -> keras.Layer:
    """
    mobilevit block

    Args:
        layer (keras.Layer): input layer
        patch_width (int): width of the patch
        patch_height (int): height of the patch
        transformer_channels (int): number of channels to use in the transformer block
        transformer_blocks (int): number of transformer blocks
        number_of_heads (int): number of heads for the attention layer
        feed_forward_network_units (int): number of units in the feed forward network
        use_l2_regularization (bool): if True apply l2 regularizer, otherwise not
        block_counter (int): a progressive counter for assigning meaningful names to basic layers
        attention_dropout_rate (float, optional): dropout rate for the attention layer. Defaults to 0.1.
        feed_forward_network_dropout_rate (float, optional): dropout rate for the feed forward network. Defaults to 0.1.
        dropout_rate (float, optional): dropout rate between attention layer and feed forward network. Defaults to 0.1.

    Returns:
        keras.Layer: output from mobilevit block
    """

    # name prefix for basic layers    
    name_prefix = f'block{str(block_counter).zfill(2)}-mvit'

    # save input channels and input layer
    _, input_height, input_width, input_channels = layer.shape
    input_layer = layer

    # l2 regularizer
    if use_l2_regularization:
        l2_regularizer = keras.regularizers.L2()
    else:
        l2_regularizer = None    

    # local representation
    layer = keras.layers.Conv2D(filters=input_channels, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-lrep-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-lrep-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-lrep-silu')(layer)
    layer = keras.layers.Conv2D(filters=transformer_channels, kernel_size=1, use_bias=False, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-lrep-exp-conv')(layer)

    # unfold
    patch_area = int(patch_width * patch_height)
    number_of_patches = int(input_width * input_height / patch_area)
    layer = keras.layers.Reshape(target_shape=(patch_area, number_of_patches, transformer_channels), name=f'{name_prefix}-unfold')(layer)

    # sequence of transformer blocks
    for n in range(transformer_blocks):
        layer = transformer_block(
            layer=layer,
            number_of_heads=number_of_heads,
            transformer_channels=transformer_channels,
            attention_dropout_rate=attention_dropout_rate,
            dropout_rate=dropout_rate,
            feed_forward_network_units=feed_forward_network_units,
            feed_forward_dropout_rate=feed_forward_network_dropout_rate,
            use_l2_regularization=use_l2_regularization,
            name_prefix=f'{name_prefix}-t{str(n).zfill(2)}'
        )

    # fold
    layer = keras.layers.Reshape(target_shape=(input_width, input_height, transformer_channels), name=f'{name_prefix}-fold')(layer)

    # fusion - reduce channels to input channels with pointwise convolution
    layer = keras.layers.Conv2D(filters=input_channels, kernel_size=1, use_bias=False, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-fus-proj-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-fus-proj-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-fus-proj-silu')(layer)

    # fusion - concatenate with input layer
    layer = keras.layers.Concatenate(axis=-1, name=f'{name_prefix}-fus-concat')([layer, input_layer])
    
    # fusion - reduce channels to input channels with standard convolution
    layer = keras.layers.Conv2D(filters=input_channels, kernel_size=3, padding='same', use_bias=False, kernel_regularizer=l2_regularizer, name=f'{name_prefix}-fus-conv')(layer)
    layer = keras.layers.BatchNormalization(name=f'{name_prefix}-fus-bnorm')(layer)
    layer = keras.layers.Activation(activation='silu', name=f'{name_prefix}-fus-silu')(layer)    

    return layer
