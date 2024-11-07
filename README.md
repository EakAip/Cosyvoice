# 声音克隆

## 创建 Conda 环境：

```python
conda create -n cosyvoice python=3.8
conda activate cosyvoice
```

```python
conda install -y -c conda-forge pynini==2.1.5
```
```python
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```
```python
sudo apt-get install sox libsox-dev
```


## 模型下载，请确保已安装git lfs
```python
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-25Hz.git pretrained_models/CosyVoice-300M-25Hz
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
git clone https://www.modelscope.cn/iic/CosyVoice-300M-Instruct.git pretrained_models/CosyVoice-300M-Instruct
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```
## 安装好目录格式如下

pretrained_models

├── CosyVoice-300M

├── CosyVoice-300M-Instruct

├── CosyVoice-300M-SFT

└── CosyVoice-ttsfrd



## 解压缩资源并安装包，以获得更好的文本规范化性能。ttsfrdttsfrd

cd pretrained_models/CosyVoice-ttsfrd/

unzip resource.zip -d .

pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
