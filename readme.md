# UNet

This project is a Python-based implementation of the U-Net architecture for image segmentation tasks. The U-Net model is a type of convolutional neural network that is widely used for biomedical image segmentation, achieving state-of-the-art results on a wide range of datasets.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python
- PyTorch
- torchvision
- tqdm

### Installing

Clone the repository to your local machine.

```bash
git clone <repository_link>
```

Install the required packages.

```bash
pip install -r requirements.txt
```

### Usage

Train the model using the following command:

```bash
python train.py
```

## Project Structure

- `model.py`: This file defines the U-Net model structure. The U-Net model is a fully convolutional network, consisting of an encoder (downsampling) part and a decoder (upsampling) part.
- `data.py`: This file defines a `MyDataset` class for loading and processing image data. This class inherits from `torch.utils.data.Dataset` and implements the `__getitem__` and `__len__` methods.
- `utils.py`: This file defines some helper functions for image processing.
- `train.py`: This file defines the training process of the model. During training, the dataset is loaded, the model and optimizer are created, and for each batch of data, forward propagation is performed, the loss is calculated, backpropagation is performed, and the model parameters are updated.

## Built With

- [PyTorch](https://pytorch.org/) - The deep learning framework used
- [torchvision](https://pytorch.org/vision/stable/index.html) - Used to handle image data
- [tqdm](https://tqdm.github.io/) - Used to display progress bars

## Authors

- Yan Qian