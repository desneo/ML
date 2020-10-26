# 1.conda安装
0) [安装cuda](https://blog.csdn.net/weixin_43318717/article/details/94433790)
    nvcc -V  //查看cuda版本
1) [下载conda 选择一个版本](https://www.anaconda.com/products/individual)
2) 创建conda环境：
       conda create -n pytorch_1027_cpuonly python=3.7   或 conda create -n pytorch_1027_GPU python=3.7
    切换到创建的环境： conda activate pytorch_1027_GPU
3) [安装pytorch](https://pytorch.org/)
       conda install pytorch torchvision cpuonly -c pytorch
    或  conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
       


# 2.Conda操作
```
1 conda update -n base conda #update最新版本的conda
2 conda create -n pytorch_1027_cpuonly python=3.7 #创建python3.7的xxxx虚拟环境
3 conda activate pytorch_1027_GPU #开启xxxx环境
4 conda deactivate #关闭环境
5 conda env list #显示所有的虚拟环境
6 conda info --envs #显示所有的虚拟环境
```