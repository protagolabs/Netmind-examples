# we assume you already the installation steps in conda_install.sh 

conda update --force conda

conda create --name netmind-tf python=3.8

source ~/miniconda3/bin/activate

conda activate netmind-tf

conda install -c anaconda cudnn==8.2.1 cudatoolkit==11.3.1

pip install tensorflow-gpu==2.9.1