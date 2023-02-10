# Netmind-AI
<div id="top" align="left">

   Netmind-AI: An open platform for centralized and decentralized deep learning training & inference

</div>

## Latest News
* [2023/01] [Netmind websited online](https://private-web.protago-dev.com/#/home)


## Why Netmind-AI
<div align="center">

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

### GPT-2
<p align="center">
(https://arxiv.org/abs/2104.05343)
</p>

- Save 50% GPU resources, and 10.7% acceleration

### GPT-2
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2.png" width=800/>

- 11x lower GPU memory consumption, and superlinear scaling efficiency with Tensor Parallelism

<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/(updated)GPT-2.png" width=800>

- 24x larger model size on the same hardware
- over 3x acceleration
### BERT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BERT.png" width=800/>

- 2x faster training, or 50% longer sequence length

### PaLM
- [PaLM-colossalai](https://github.com/hpcaitech/PaLM-colossalai): Scalable implementation of Google's Pathways Language Model ([PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)).

### OPT
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_update.png" width=800/>

- [Open Pretrained Transformer (OPT)](https://github.com/facebookresearch/metaseq), a 175-Billion parameter AI language model released by Meta, which stimulates AI programmers to perform various downstream tasks and application deployments because public pretrained model weights.
- 45% speedup fine-tuning OPT at low cost in lines. [[Example]](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/language/opt) [[Online Serving]](https://github.com/hpcaitech/ColossalAI-Documentation/blob/main/i18n/en/docusaurus-plugin-content-docs/current/advanced_tutorials/opt_service.md)

Please visit our [documentation](https://www.colossalai.org/) and [examples](https://github.com/hpcaitech/ColossalAI-Examples) for more details.

### ViT
<p align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/ViT.png" width="450" />
</p>

- 14x larger batch size, and 5x faster training for Tensor Parallelism = 64

### Recommendation System Models
- [Cached Embedding](https://github.com/hpcaitech/CachedEmbedding), utilize software cache to train larger embedding tables with a smaller GPU memory budget.

<p align="right">(<a href="#top">back to top</a>)</p>

## Single GPU Training Demo

### GPT-2
<p id="GPT-2-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-GPU1.png" width=450/>
</p>

- 20x larger model size on the same hardware

<p id="GPT-2-NVME" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/GPT2-NVME.png" width=800/>
</p>

- 120x larger model size on the same hardware (RTX 3080)

### PaLM
<p id="PaLM-Single" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/PaLM-GPU1.png" width=450/>
</p>

- 34x larger model size on the same hardware

<p align="right">(<a href="#top">back to top</a>)</p>


## Inference (Energon-AI) Demo

<p id="GPT-3-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/inference_GPT-3.jpg" width=800/>
</p>

- [Energon-AI](https://github.com/hpcaitech/EnergonAI): 50% inference acceleration on the same hardware

<p id="OPT-Serving" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/OPT_serving.png" width=800/>
</p>

- [OPT Serving](https://github.com/hpcaitech/ColossalAI-Documentation/blob/main/i18n/en/docusaurus-plugin-content-docs/current/advanced_tutorials/opt_service.md): Try 175-billion-parameter OPT online services for free, without any registration whatsoever.

<p id="BLOOM-Inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/BLOOM%20Inference.PNG" width=800/>
</p>

- [BLOOM](https://github.com/hpcaitech/EnergonAI/tree/main/examples/bloom): Reduce hardware deployment costs of 176-billion-parameter BLOOM by more than 10 times.

<p align="right">(<a href="#top">back to top</a>)</p>

## Colossal-AI in the Real World

### AIGC
Acceleration of AIGC (AI-Generated Content) models such as [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion) and [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion).
<p id="diffusion_train" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20v2.png" width=800/>
</p>

- [Training](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): Reduce Stable Diffusion memory consumption by up to 5.6x and hardware cost by up to 46x (from A100 to RTX3060).

<p id="diffusion_demo" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/DreamBooth.png" width=800/>
</p>

- [DreamBooth Fine-tuning](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/dreambooth): Personalize your model using just 3-5 images of the desired subject.

<p id="inference" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/Stable%20Diffusion%20Inference.jpg" width=800/>
</p>

- [Inference](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion): Reduce inference GPU memory consumption by 2.5x.


<p align="right">(<a href="#top">back to top</a>)</p>

### Biomedicine
Acceleration of [AlphaFold Protein Structure](https://alphafold.ebi.ac.uk/)

<p id="FastFold" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/FastFold.jpg" width=800/>
</p>

- [FastFold](https://github.com/hpcaitech/FastFold): accelerating training and inference on GPU Clusters, faster data processing, inference sequence containing more than 10000 residues.

<p id="xTrimoMultimer" align="center">
<img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/xTrimoMultimer_Table.jpg" width=800/>
</p>

- [xTrimoMultimer](https://github.com/biomap-research/xTrimoMultimer): accelerating structure prediction of protein monomers and multimer by 11x.


<p align="right">(<a href="#top">back to top</a>)</p>

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


