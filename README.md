# Introduction
I believe that one of the most interesting aspects of deep learning is the design of network architectures for computer vision applications.

Among all available architectures, mobile networks have always intrigued me the most. Designing an effective network architecture and achieving good results is challenging, even for large networks with a high number of parameters. Achieving good results with small networks, especially those intended for deployment on mobile or low-powered devices, is even more difficult. These networks face constraints on size, operation complexity, and typically require high throughput rates. Mobile networks aim to address these issues while still delivering performance close to that of larger networks!

# "Light-weight, General-purpose, and Mobile-friendly Vision Transformer" - MobileViT Network
Last year, I read the paper ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/pdf/2110.02178) and found it to very interesting.

I really wanted to code the network from scratch to have a better understanding, but unfortunately I never found the time to do it!

Now I finally found it!

## Transformers vs CNNs

Transformers have become the go-to for natural language tasks and are now showing up in computer vision, though they haven’t replaced traditional CNNs. While Transformers score high on vision benchmarks, they’re parameter-heavy, slow, and have complex training processes.

CNNs, on the other hand, are lighter on parameters, faster, and easier to train, but they don’t quite match the precision of Transformers.

So, why are these architectures so different?

### Transformers vs CNNs - "Different Vision"

CNNs rely on convolutions, where filters with small kernels slide over image data. Typically, CNNs start by reducing the image size while increasing the number of channels. Early layers capture fine details, while later layers focus on larger patterns like edges or shapes. A small 3x3 kernel on a 256x256 image processes tiny areas, but on an 8x8 image, it covers much more. As the image shrinks, the increased number of channels helps retain information.

This means CNNs have a limited "field of view," focusing on local details. Pyramidal designs help by shrinking the image and expanding the receptive field. Another approach is dilated convolution, which increases the receptive field without reducing resolution, though finding the right dilation rate can be tricky.

Transformers, on the other hand, work differently. They don't learn local representations but instead have a full "field of view" thanks to self-attention, allowing them to learn global representations. This is why Transformers achieve higher scores, but they require a lot of training data and complex augmentation to avoid overfitting. With limited data, Transformers risk poor generalization!

### Mobile-friendly Vision Transformer
Well.. high number of parameters, complex operations and slow througput rate sound like the worst possible recipe for a network that needs to work on a mobile low-powered device for real-time operations.

Does this mean that's not possible to design small lightweight transformers based network? Not quite.

MobileViT it's an interesting architecture that combines MobileNet-v2 basic building blocks with transformer based building blocks. It's an hybrid architecture which have both advantages of CNNs (learning local represantion) and Transformers (learning global representations). Thanks to the CNNs components MobileViT can be trained with typical and simple data augmentation and training procedures.

### MobileViT - The Downside
MobileViT is an interesting network architecture that combines the strengths of both CNNs and Transformers. This enables simpler and more effective training procedures compared to pure Transformers and often results in slightly better performance than most CNNs with a similar number of parameters.

However, one drawback of MobileViT is its throughput rate. While it performs well, it's inevitably slower than similar-sized CNNs, due to expensive multi-head attention block operations. This lower throughput could be a concern when deploying the network on mobile or low-power devices.

I've just started reading the MobileViT-v2 paper, which promises to address this issue by proposing an alternative block to the classic multi-head attention block. Maybe i'll try to implement also this variant!

# Implementation from Scratch
You can check out the *mobilevitlib* folder in this repository, which contains a simple Python custom module. The code uses standard Python type hints and docstrings, aiming to keep things straightforward. Note that I intentionally minimized the number of functions to make the building blocks easier to understand and read (see note below).

Here’s a brief overview of each file in the *mobilevitlib* module:
- **blocks.py** -> the mobilevit basic building block
- **models.py** -> the mobilevit network architecture
- **plots.py** -> simple function for plotting models training history (train and validation loss)
- **processing.py** -> functions for data augmentation and data encoding

You can use alternatively the notebook "*mobilevit-flowers-102-experiment.ipynb*" or the script "*mobilevit-flowers-102-experiment.py*" to run an experiment on flowers.102 data using a mobilevit network!

You can install the required dependencies using the "*requirements.txt*" file (WARNING -> jupyter notebooks dependencies not included), everything should work as expected using python 3.12 or similar versions. I tested everything on WSL2 with standard Ubuntu distribution.

**NOTE**: in this implementation i deliberately tried to use as few functions as possible. While it's useful to encapsulate repetetive basic building blocks (ex. convolution, batchnorm, activation) in functions, it quickly became a real nightmare when the only goal it's to understand a network design. When I look for network implementations and my goal is pure architectures understanding, I personally hate the functions inceptions when building simple stuff. Hope you will find easier to read the implementation in this way!
