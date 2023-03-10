We develop the netmind platform in tensorflow2

# install conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

conda update --force conda
```
#

# install the tensorflow and some other deps
* create netmind-tf env
```bash
conda create --name netmind-tf python=3.8
```
* load enviroment
```bash
conda activate netmind-tf
```

* install the cuda and cudnn
```bash
conda install -c anaconda cudnn==8.2.1 cudatoolkit==11.3.1
```

* add cuda/cudnn to path, otherwise you will meet issue when running tensorflow as: Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
```bash
nano ~/.bashrc
```
* add the following cuda/cudnn path to the end, change `YOUR_HOME_DIR` to your home directory
```
# change YOUR_HOME_DIR to your home directory
export YOUR_HOME_DIR=/home/xxx/
export PATH=$YOUR_HOME_DIR/miniconda3/envs/netmind-tf/lib:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$YOUR_HOME_DIR/miniconda3/envs/netmind-tf/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

* load again
```bash
source ~/.bashrc
```
```bash
conda activate netmind-tf
```
* install tensorflow-gpu
```bash
pip install tensorflow-gpu==2.9.1
```
#