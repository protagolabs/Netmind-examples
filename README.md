# Netmind-AI
<div id="top" align="left">

   Netmind-AI: An open platform for centralized and decentralized deep learning training & inference

</div>

## Latest News
* [2023/01] [Netmind websited online](https://private-web.protago-dev.com/#/home)


## Why Netmind-AI
<div align="left">

   * Users can train the corresponding artificial intelligence by providing algorithms and data on the Netmind platform.
   * Netmind AI platform not only includes training resources and its own distributed traing algorithms, but also support thrid-party algorithms.
   * On the Netmind platform, users can freely deploy their own models, algorithms, and share their own models or adopt others' model for communication  and comparison.
   * Users can obtain inference function services by deploying their own models on Netmind.
</div>

<p align="right">(<a href="#top">back to top</a>)</p>

## Features

Netmind-AI provides centralized and decentralized training framework for you. We aim to support you to write your
distributed deep learning models just like how you write your model on your laptop. We provide user-friendly tools to kickstart
distributed training and inference in a few lines.

- Centralized distributed traning
  - Data Parallelism
  - Model Parallelism
  - 1D, [2D](https://arxiv.org/abs/2104.05343), [2.5D](https://arxiv.org/abs/2105.14500), [3D](https://arxiv.org/abs/2105.14450) Tensor Parallelism (coming soon)
 

- Decentralized distributed traning
  - [Hivemind](https://arxiv.org/abs/2002.04013)
  - [Hivemind](https://arxiv.org/abs/2103.03239)

- Friendly Usage
  - Codelab example that easy to deploy or modified.

<p align="right">(<a href="#top">back to top</a>)</p>

## Parallel Training Demo

### Language Model
<p align="left">
[local](https://github.com/protagolabs/Netmind-examples/tree/main/pytorch/language-modeling/local)
</p>


### Resnet

<p align="left">
[local](https://github.com/protagolabs/Netmind-examples/tree/main/pytorch/resnet/local)
</p>


## Installation

### Install from PyPI

You can easily install Colossal-AI with the following command. **By defualt, we do not build PyTorch extensions during installation.**

```bash
pip install colossalai
```

However, if you want to build the PyTorch extensions during installation, you can set `CUDA_EXT=1`.

```bash
CUDA_EXT=1 pip install colossalai
```

**Otherwise, CUDA kernels will be built during runtime when you actually need it.**

We also keep release the nightly version to PyPI on a weekly basis. This allows you to access the unreleased features and bug fixes in the main branch.
Installation can be made via

```bash
pip install colossalai-nightly
```

### Download From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem. :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install colossalai
pip install .
```

By default, we do not compile CUDA/C++ kernels. ColossalAI will build them during runtime.
If you want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer):

```shell
CUDA_EXT=1 pip install .
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Use Docker

### Pull from DockerHub

You can directly pull the docker image from our [DockerHub page](https://hub.docker.com/r/hpcaitech/colossalai). The image is automatically uploaded upon release.


### Build On Your Own

Run the following command to build a docker image from Dockerfile provided.

> Building Colossal-AI from scratch requires GPU support, you need to use Nvidia Docker Runtime as the default when doing `docker build`. More details can be found [here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).
> We recommend you install Colossal-AI from our [project page](https://www.colossalai.org) directly.


```bash
cd ColossalAI
docker build -t colossalai ./docker
```

Run the following command to start the docker container in interactive mode.

```bash
docker run -ti --gpus all --rm --ipc=host colossalai bash
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Community



## Contributing


