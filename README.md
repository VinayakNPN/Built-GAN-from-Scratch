# 🧠 GAN from Scratch - MNIST Digit Generator

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=gan-from-scratch)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🚀 I built a GenAI model from scratch** - A complete implementation of Generative Adversarial Networks (GANs) for generating handwritten digits using the MNIST dataset.

## 📖 What are GANs?

**Generative Adversarial Networks (GANs)** are a revolutionary class of machine learning models introduced by Ian Goodfellow in 2014. They consist of two neural networks competing against each other in a zero-sum game framework:


### 🎭 The Two Players

#### 🎨 **Generator**
- **Role**: Creates fake data that mimics real data
- **Goal**: Fool the discriminator into thinking generated data is real
- **Input**: Random noise vector (latent space)
- **Output**: Synthetic data (in our case, 28x28 digit images)

#### 🕵️ **Discriminator**
- **Role**: Distinguishes between real and fake data
- **Goal**: Correctly classify real vs generated data
- **Input**: Real or generated images
- **Output**: Probability score (0 = fake, 1 = real)

### ⚔️ The Adversarial Process

The training process resembles a **cat-and-mouse game**:

1. **Generator** creates fake images from random noise
2. **Discriminator** evaluates both real MNIST images and generated fakes
3. Both networks improve through backpropagation:
   - Generator learns to create more realistic images
   - Discriminator becomes better at spotting fakes
4. Eventually, the Generator becomes so good that the Discriminator can't tell the difference!

![GAN Training Process](https://developers.google.com/static/machine-learning/gan/images/gan_diagram.svg)

## 🏗️ Architecture Details

### Generator Network
```
Input: 100D noise vector
├── Dense(256) + LeakyReLU + BatchNorm
├── Dense(512) + LeakyReLU + BatchNorm  
├── Dense(1024) + LeakyReLU + BatchNorm
├── Dense(784) + Tanh
└── Reshape(28, 28, 1)
```

### Discriminator Network
```
Input: 28x28x1 image
├── Flatten(784)
├── Dense(512) + LeakyReLU
├── Dense(256) + LeakyReLU
└── Dense(1) + Sigmoid
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy matplotlib
```


### 🎯 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gan-from-scratch.git
cd gan-from-scratch
```

2. **Install dependencies**
```bash
pip install tensorflow numpy matplotlib
```

3. **Run the notebook**
```bash
jupyter notebook gan.ipynb
```

4. **Watch the magic happen!** ✨
   - The model will start generating images every 10 epochs
   - Images are saved as `gan_images_epoch_{epoch}.png`

## 📊 Training Process

The model trains for **10,000 epochs** with the following hyperparameters:
- **Batch Size**: 64
- **Learning Rate**: 0.0002 (Adam optimizer)
- **Beta1**: 0.5
- **Latent Dimension**: 100

### 📈 Loss Functions
- **Generator Loss**: Binary crossentropy (tries to fool discriminator)
- **Discriminator Loss**: Binary crossentropy (real vs fake classification)

## 🎨 Results

As training progresses, you'll see the generated digits evolve from random noise to recognizable handwritten digits:

**Epoch 0** → **Epoch 1000** → **Epoch 5000** → **Epoch 10000**

*Random noise* → *Blurry shapes* → *Digit-like forms* → *Realistic digits*

## 🔧 Key Features

- ✅ **Built from scratch** using TensorFlow/Keras
- ✅ **Well-documented** code with clear explanations
- ✅ **Visualization** of training progress
- ✅ **Modular design** for easy experimentation
- ✅ **MNIST dataset** for quick testing and validation

## 🧪 Experiments & Modifications

Want to experiment? Try these modifications:

1. **Change the latent dimension** (currently 100)
2. **Modify network architectures** (add/remove layers)
3. **Adjust hyperparameters** (learning rate, batch size)
4. **Try different datasets** (CIFAR-10, CelebA)
5. **Implement different GAN variants** (DCGAN, WGAN, etc.)

## 📚 Understanding the Code

### Key Components

1. **Data Preprocessing**: MNIST images normalized to [-1, 1] range
2. **Generator**: Transforms noise into realistic images
3. **Discriminator**: Binary classifier for real vs fake
4. **Adversarial Training**: Alternating training of both networks
5. **Visualization**: Periodic image generation for progress tracking

### 🔍 Training Loop Breakdown
```python
for epoch in range(epochs):
    # Train Discriminator
    real_images = sample_real_batch()
    fake_images = generator.predict(noise)
    d_loss = train_discriminator(real_images, fake_images)
    
    # Train Generator
    g_loss = train_generator(noise)
    
    # Visualize progress
    if epoch % 10 == 0:
        generate_and_save_images()
```


## 🙏 Acknowledgments

- **Ian Goodfellow** for inventing GANs
- **TensorFlow team** for the amazing framework
- **MNIST dataset** creators for the classic benchmark
- **Open source community** for inspiration and resources


**⭐ Star this repo if you found it helpful!**

*Built with ❤️ and by Vinayak Chouhan ☕*
