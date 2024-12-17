{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "2ce00cc7",
   "metadata": {},
   "source": [
    "## Install Sagemaker SDK, Transformers in Python3:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "638e21f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Looking in indexes: https://pypi.org/simple, https://pip.repos.neuron.amazonaws.com\n",
      "Requirement already satisfied: torch-neuron in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (1.10.2.2.3.0.0)\n",
      "Collecting torch-neuron\n",
      "  Downloading https://pip.repos.neuron.amazonaws.com/torch-neuron/torch_neuron-1.11.0.2.3.0.0-py3-none-linux_x86_64.whl (32.6 MB)\n",
      "     |████████████████████████████████| 32.6 MB 94.1 MB/s            \n",
      "\u001b[?25hRequirement already satisfied: neuron-cc[tensorflow] in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (1.11.4.0+97f99abe4)\n",
      "Requirement already satisfied: torchvision in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (0.3.0)\n",
      "Collecting torchvision\n",
      "  Downloading torchvision-0.11.2-cp36-cp36m-manylinux1_x86_64.whl (23.3 MB)\n",
      "     |████████████████████████████████| 23.3 MB 5.8 MB/s            \n",
      "\u001b[?25hRequirement already satisfied: torch in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (1.10.2)\n",
      "Requirement already satisfied: dataclasses in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from torch) (0.7)\n",
      "Requirement already satisfied: typing-extensions in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from torch) (4.0.1)\n",
      "Requirement already satisfied: islpy==2021.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (2021.1+aws2021.x.16.0.bld0)\n",
      "Requirement already satisfied: dmlc-topi==1.11.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.11.0.0+0)\n",
      "Requirement already satisfied: networkx<=2.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (2.4)\n",
      "Requirement already satisfied: numpy<=1.20.0,>=1.13.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.19.5)\n",
      "Requirement already satisfied: scipy<=1.4.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.4.1)\n",
      "Requirement already satisfied: inferentia-hwm==1.11.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.11.0.0+0)\n",
      "Requirement already satisfied: dmlc-tvm==1.11.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.11.0.0+0)\n",
      "Requirement already satisfied: dmlc-nnvm==1.11.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.11.0.0+0)\n",
      "Requirement already satisfied: tensorflow<=1.15 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from neuron-cc[tensorflow]) (1.15.0)\n",
      "Requirement already satisfied: decorator in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from dmlc-topi==1.11.0.0->neuron-cc[tensorflow]) (4.4.2)\n",
      "Requirement already satisfied: attrs in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from dmlc-tvm==1.11.0.0->neuron-cc[tensorflow]) (20.3.0)\n",
      "Requirement already satisfied: pytest>=2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from islpy==2021.1->neuron-cc[tensorflow]) (6.2.2)\n",
      "Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from islpy==2021.1->neuron-cc[tensorflow]) (1.15.0)\n",
      "Requirement already satisfied: pillow!=8.3.0,>=5.3.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from torchvision) (8.4.0)\n",
      "  Downloading torchvision-0.11.1-cp36-cp36m-manylinux1_x86_64.whl (23.3 MB)\n",
      "     |████████████████████████████████| 23.3 MB 37.2 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.10.1-cp36-cp36m-manylinux1_x86_64.whl (22.1 MB)\n",
      "     |████████████████████████████████| 22.1 MB 23.0 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.10.0-cp36-cp36m-manylinux1_x86_64.whl (22.1 MB)\n",
      "     |████████████████████████████████| 22.1 MB 26.7 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.9.1-cp36-cp36m-manylinux1_x86_64.whl (17.4 MB)\n",
      "     |████████████████████████████████| 17.4 MB 25.8 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.9.0-cp36-cp36m-manylinux1_x86_64.whl (17.3 MB)\n",
      "     |████████████████████████████████| 17.3 MB 33.5 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.8.2-cp36-cp36m-manylinux1_x86_64.whl (12.8 MB)\n",
      "     |████████████████████████████████| 12.8 MB 23.9 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.8.1-cp36-cp36m-manylinux1_x86_64.whl (12.8 MB)\n",
      "     |████████████████████████████████| 12.8 MB 32.9 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.8.0-cp36-cp36m-manylinux1_x86_64.whl (11.8 MB)\n",
      "     |████████████████████████████████| 11.8 MB 35.3 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.7.0-cp36-cp36m-manylinux1_x86_64.whl (5.9 MB)\n",
      "     |████████████████████████████████| 5.9 MB 40.1 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.6.1-cp36-cp36m-manylinux1_x86_64.whl (6.6 MB)\n",
      "     |████████████████████████████████| 6.6 MB 38.2 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.6.0-cp36-cp36m-manylinux1_x86_64.whl (6.6 MB)\n",
      "     |████████████████████████████████| 6.6 MB 22.1 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.5.0-cp36-cp36m-manylinux1_x86_64.whl (4.0 MB)\n",
      "     |████████████████████████████████| 4.0 MB 53.5 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.4.2-cp36-cp36m-manylinux1_x86_64.whl (10.2 MB)\n",
      "     |████████████████████████████████| 10.2 MB 23.7 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.4.1-cp36-cp36m-manylinux1_x86_64.whl (10.1 MB)\n",
      "     |████████████████████████████████| 10.1 MB 22.4 MB/s            \n",
      "\u001b[?25h  Downloading torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl (8.8 MB)\n",
      "     |████████████████████████████████| 8.8 MB 74.6 MB/s            \n",
      "\u001b[?25hRequirement already satisfied: keras-applications>=1.0.8 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.0.8)\n",
      "Requirement already satisfied: keras-preprocessing>=1.0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.1.2)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.1.0)\n",
      "Requirement already satisfied: protobuf>=3.6.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (3.15.2)\n",
      "Requirement already satisfied: tensorboard<1.16.0,>=1.15.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.15.0)\n",
      "Requirement already satisfied: tensorflow-estimator==1.15.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.15.1)\n",
      "Requirement already satisfied: wrapt>=1.11.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.12.1)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (3.3.0)\n",
      "Requirement already satisfied: astor>=0.6.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (0.8.1)\n",
      "Requirement already satisfied: google-pasta>=0.1.6 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (0.2.0)\n",
      "Requirement already satisfied: grpcio>=1.8.6 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.44.0)\n",
      "Requirement already satisfied: wheel>=0.26 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (0.36.2)\n",
      "Requirement already satisfied: gast==0.2.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (0.2.2)\n",
      "Requirement already satisfied: absl-py>=0.7.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorflow<=1.15->neuron-cc[tensorflow]) (1.0.0)\n",
      "Requirement already satisfied: h5py in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from keras-applications>=1.0.8->tensorflow<=1.15->neuron-cc[tensorflow]) (3.1.0)\n",
      "Requirement already satisfied: iniconfig in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (1.1.1)\n",
      "Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (20.9)\n",
      "Requirement already satisfied: pluggy<1.0.0a1,>=0.12 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (0.13.1)\n",
      "Requirement already satisfied: py>=1.8.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (1.10.0)\n",
      "Requirement already satisfied: toml in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (0.10.2)\n",
      "Requirement already satisfied: importlib-metadata>=0.12 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (4.8.3)\n",
      "Requirement already satisfied: werkzeug>=0.11.15 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow<=1.15->neuron-cc[tensorflow]) (1.0.1)\n",
      "Requirement already satisfied: markdown>=2.6.8 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow<=1.15->neuron-cc[tensorflow]) (3.3.6)\n",
      "Requirement already satisfied: setuptools>=41.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from tensorboard<1.16.0,>=1.15.0->tensorflow<=1.15->neuron-cc[tensorflow]) (49.6.0.post20210108)\n",
      "Requirement already satisfied: zipp>=0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata>=0.12->pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (3.4.0)\n",
      "Requirement already satisfied: cached-property in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from h5py->keras-applications>=1.0.8->tensorflow<=1.15->neuron-cc[tensorflow]) (1.5.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from packaging->pytest>=2->islpy==2021.1->neuron-cc[tensorflow]) (2.4.7)\n"
     ]
    }
   ],
   "source": [
    "!pip install --upgrade --no-cache-dir torch-neuron neuron-cc[tensorflow] torchvision torch --extra-index-url=https://pip.repos.neuron.amazonaws.com"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "49045054",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (4.12.3)\n",
      "Requirement already satisfied: dataclasses in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (0.7)\n",
      "Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (2.26.0)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (0.4.0)\n",
      "Requirement already satisfied: pyyaml>=5.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (5.4.1)\n",
      "Requirement already satisfied: importlib-metadata in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (4.8.3)\n",
      "Requirement already satisfied: tqdm>=4.27 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (4.62.3)\n",
      "Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (0.10.3)\n",
      "Requirement already satisfied: numpy>=1.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (1.19.5)\n",
      "Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (3.0.12)\n",
      "Requirement already satisfied: sacremoses in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (0.0.49)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (20.9)\n",
      "Requirement already satisfied: regex!=2019.12.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers) (2020.11.13)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from huggingface-hub<1.0,>=0.1.0->transformers) (4.0.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from packaging>=20.0->transformers) (2.4.7)\n",
      "Requirement already satisfied: zipp>=0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata->transformers) (3.4.0)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers) (1.26.8)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers) (2.0.12)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers) (2021.5.30)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers) (3.1)\n",
      "Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sacremoses->transformers) (1.15.0)\n",
      "Requirement already satisfied: click in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sacremoses->transformers) (7.1.2)\n",
      "Requirement already satisfied: joblib in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sacremoses->transformers) (1.0.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "11db3a36",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: sagemaker>=2.48.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (2.88.1)\n",
      "Requirement already satisfied: smdebug-rulesconfig==1.0.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (1.0.1)\n",
      "Requirement already satisfied: protobuf3-to-dict>=0.1.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (0.1.5)\n",
      "Requirement already satisfied: importlib-metadata>=1.4.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (4.8.3)\n",
      "Requirement already satisfied: attrs==20.3.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (20.3.0)\n",
      "Requirement already satisfied: pandas in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (1.1.5)\n",
      "Requirement already satisfied: google-pasta in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (0.2.0)\n",
      "Requirement already satisfied: boto3>=1.20.21 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (1.21.42)\n",
      "Requirement already satisfied: protobuf>=3.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (3.15.2)\n",
      "Requirement already satisfied: pathos in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (0.2.8)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (20.9)\n",
      "Requirement already satisfied: numpy>=1.9.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sagemaker>=2.48.0) (1.19.5)\n",
      "Requirement already satisfied: botocore<1.25.0,>=1.24.42 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3>=1.20.21->sagemaker>=2.48.0) (1.24.42)\n",
      "Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3>=1.20.21->sagemaker>=2.48.0) (0.10.0)\n",
      "Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3>=1.20.21->sagemaker>=2.48.0) (0.5.0)\n",
      "Requirement already satisfied: zipp>=0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata>=1.4.0->sagemaker>=2.48.0) (3.4.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata>=1.4.0->sagemaker>=2.48.0) (4.0.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from packaging>=20.0->sagemaker>=2.48.0) (2.4.7)\n",
      "Requirement already satisfied: six>=1.9 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from protobuf>=3.1->sagemaker>=2.48.0) (1.15.0)\n",
      "Requirement already satisfied: python-dateutil>=2.7.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas->sagemaker>=2.48.0) (2.8.1)\n",
      "Requirement already satisfied: pytz>=2017.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas->sagemaker>=2.48.0) (2021.1)\n",
      "Requirement already satisfied: pox>=0.3.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.48.0) (0.3.0)\n",
      "Requirement already satisfied: multiprocess>=0.70.12 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.48.0) (0.70.12.2)\n",
      "Requirement already satisfied: dill>=0.3.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.48.0) (0.3.4)\n",
      "Requirement already satisfied: ppft>=1.6.6.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pathos->sagemaker>=2.48.0) (1.6.6.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.25.4 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from botocore<1.25.0,>=1.24.42->boto3>=1.20.21->sagemaker>=2.48.0) (1.26.8)\n"
     ]
    }
   ],
   "source": [
    "# ! pip install sagemaker --upgrade\n",
    "!pip install \"sagemaker>=2.48.0\" --upgrade"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6dcd5de5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers==4.12.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (4.12.3)\n",
      "Requirement already satisfied: requests in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (2.26.0)\n",
      "Requirement already satisfied: numpy>=1.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (1.19.5)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (0.4.0)\n",
      "Requirement already satisfied: regex!=2019.12.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (2020.11.13)\n",
      "Requirement already satisfied: tqdm>=4.27 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (4.62.3)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (20.9)\n",
      "Requirement already satisfied: importlib-metadata in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (4.8.3)\n",
      "Requirement already satisfied: sacremoses in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (0.0.49)\n",
      "Requirement already satisfied: dataclasses in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (0.7)\n",
      "Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (3.0.12)\n",
      "Requirement already satisfied: pyyaml>=5.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (5.4.1)\n",
      "Requirement already satisfied: tokenizers<0.11,>=0.10.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from transformers==4.12.3) (0.10.3)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from huggingface-hub<1.0,>=0.1.0->transformers==4.12.3) (4.0.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from packaging>=20.0->transformers==4.12.3) (2.4.7)\n",
      "Requirement already satisfied: zipp>=0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata->transformers==4.12.3) (3.4.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers==4.12.3) (3.1)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers==4.12.3) (2.0.12)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers==4.12.3) (1.26.8)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests->transformers==4.12.3) (2021.5.30)\n",
      "Requirement already satisfied: click in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sacremoses->transformers==4.12.3) (7.1.2)\n",
      "Requirement already satisfied: six in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sacremoses->transformers==4.12.3) (1.15.0)\n",
      "Requirement already satisfied: joblib in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from sacremoses->transformers==4.12.3) (1.0.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install \"transformers==4.12.3\" --upgrade --no-cache-dir"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7fa9aafa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: datasets[s3]==1.18.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (1.18.3)\n",
      "Requirement already satisfied: multiprocess in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (0.70.12.2)\n",
      "Requirement already satisfied: dill in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (0.3.4)\n",
      "Requirement already satisfied: importlib-metadata in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (4.8.3)\n",
      "Requirement already satisfied: packaging in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (20.9)\n",
      "Requirement already satisfied: fsspec[http]>=2021.05.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (2022.1.0)\n",
      "Requirement already satisfied: pyarrow!=4.0.0,>=3.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (6.0.1)\n",
      "Requirement already satisfied: tqdm>=4.62.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (4.62.3)\n",
      "Requirement already satisfied: requests>=2.19.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (2.26.0)\n",
      "Requirement already satisfied: numpy>=1.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (1.19.5)\n",
      "Requirement already satisfied: pandas in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (1.1.5)\n",
      "Requirement already satisfied: aiohttp in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (3.8.1)\n",
      "Requirement already satisfied: huggingface-hub<1.0.0,>=0.1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (0.4.0)\n",
      "Requirement already satisfied: dataclasses in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (0.7)\n",
      "Requirement already satisfied: xxhash in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (3.0.0)\n",
      "Requirement already satisfied: botocore in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (1.24.42)\n",
      "Requirement already satisfied: boto3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (1.21.42)\n",
      "Requirement already satisfied: s3fs in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from datasets[s3]==1.18.3) (0.4.2)\n",
      "Requirement already satisfied: filelock in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets[s3]==1.18.3) (3.0.12)\n",
      "Requirement already satisfied: pyyaml in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets[s3]==1.18.3) (5.4.1)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from huggingface-hub<1.0.0,>=0.1.0->datasets[s3]==1.18.3) (4.0.1)\n",
      "Requirement already satisfied: pyparsing>=2.0.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from packaging->datasets[s3]==1.18.3) (2.4.7)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests>=2.19.0->datasets[s3]==1.18.3) (3.1)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests>=2.19.0->datasets[s3]==1.18.3) (1.26.8)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests>=2.19.0->datasets[s3]==1.18.3) (2.0.12)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from requests>=2.19.0->datasets[s3]==1.18.3) (2021.5.30)\n",
      "Requirement already satisfied: asynctest==0.13.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (0.13.0)\n",
      "Requirement already satisfied: attrs>=17.3.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (20.3.0)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (1.2.0)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (1.6.3)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (5.1.0)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (1.2.0)\n",
      "Requirement already satisfied: idna-ssl>=1.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (1.1.0)\n",
      "Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from aiohttp->datasets[s3]==1.18.3) (4.0.1)\n",
      "Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3->datasets[s3]==1.18.3) (0.5.0)\n",
      "Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from boto3->datasets[s3]==1.18.3) (0.10.0)\n",
      "Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from botocore->datasets[s3]==1.18.3) (2.8.1)\n",
      "Requirement already satisfied: zipp>=0.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from importlib-metadata->datasets[s3]==1.18.3) (3.4.0)\n",
      "Requirement already satisfied: pytz>=2017.2 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from pandas->datasets[s3]==1.18.3) (2021.1)\n",
      "Requirement already satisfied: six>=1.5 in /home/ec2-user/anaconda3/envs/python3/lib/python3.6/site-packages (from python-dateutil<3.0.0,>=2.1->botocore->datasets[s3]==1.18.3) (1.15.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install \"datasets[s3]==1.18.3\" --upgrade"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "496e3492",
   "metadata": {},
   "source": [
    "## Install Libraries:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4643e6d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "transformers.__version__: 4.12.3\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch_neuron\n",
    "import transformers\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig\n",
    "\n",
    "\n",
    "print(\"transformers.__version__:\", transformers.__version__)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d386230",
   "metadata": {},
   "source": [
    "## Create SageMaker endpoint with the chosen model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "97738e07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sagemaker_session_bucket: sagemaker-us-west-2-893195739154\n",
      "role: arn:aws:iam::893195739154:role/service-role/AmazonSageMaker-ExecutionRole-20220422T170187\n",
      "sagemaker role arn: arn:aws:iam::893195739154:role/service-role/AmazonSageMaker-ExecutionRole-20220422T170187\n",
      "sagemaker bucket: sagemaker-us-west-2-893195739154\n",
      "sagemaker session region: us-west-2\n"
     ]
    }
   ],
   "source": [
    "import sagemaker\n",
    "from sagemaker import get_execution_role\n",
    "import json\n",
    "import boto3\n",
    "\n",
    "\n",
    "sess = sagemaker.Session()\n",
    "\n",
    "\n",
    "\n",
    "sagemaker_session_bucket= None # \"sagemaker-us-west-2-893195739154\"\n",
    "if sagemaker_session_bucket is None and sess is not None:\n",
    "    pass\n",
    "    sagemaker_session_bucket = sess.default_bucket() # set to default bucket if a bucket name is not given\n",
    "\n",
    "print(\"sagemaker_session_bucket:\", sagemaker_session_bucket)\n",
    "\n",
    "\n",
    "role = get_execution_role()\n",
    "print(\"role:\", role)\n",
    "\n",
    "\n",
    "print(f\"sagemaker role arn: {role}\")\n",
    "print(f\"sagemaker bucket: {sess.default_bucket()}\")\n",
    "print(f\"sagemaker session region: {sess.boto_region_name}\")\n",
    "\n",
    "\n",
    "prefix = \"mymodel/supervised\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "edae2a46",
   "metadata": {},
   "outputs": [],
   "source": [
    "aws_region = \"us-west-2\" # AWS-region"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d70a592b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "img_uri: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.10-gpu-py38\n"
     ]
    }
   ],
   "source": [
    "img_uri = sagemaker.image_uris.retrieve(framework='pytorch', \n",
    "            region=aws_region, \n",
    "            image_scope='inference', \n",
    "            version=\"1.10\", \n",
    "            instance_type='ml.p3.16xlarge', \n",
    "            py_version='py38') \n",
    "print(\"img_uri:\", img_uri)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "697b6ca5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "from sagemaker.huggingface.model import HuggingFaceModel\n",
    "\n",
    "\n",
    "hub = {\n",
    "\n",
    "  'HF_MODEL_ID':'distilbert-base-uncased-distilled-squad', # model_id from hf.co/models\n",
    "  'HF_TASK':'question-answering'\n",
    "}\n",
    "\n",
    "\n",
    "pytorch_cpu_image_uri = \"763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04\"\n",
    "\n",
    "\n",
    "huggingface_model = HuggingFaceModel(\n",
    "   env=hub, \n",
    "   role = sagemaker.get_execution_role(),               \n",
    "   \n",
    "   image_uri = pytorch_cpu_image_uri,                                            \n",
    "   transformers_version=\"4.18\",                         \n",
    "   pytorch_version=\"1.7\",                                \n",
    ")\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "006f3b00",
   "metadata": {},
   "source": [
    "##  Deploy model to SageMaker Inference:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09a2b512",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "predictor = huggingface_model.deploy(\n",
    "   initial_instance_count=1,\n",
    "   instance_type=\"ml.m5.xlarge\"\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8171bc4",
   "metadata": {},
   "source": [
    "## Sample Input: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3cbbee32",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {\n",
    "\n",
    "    \"inputs\": {\n",
    "        \"question\": \"Why did Saifur stay at home?\",\n",
    "        \"context\": \"Anna friend forgot to bring Saifur's umbrella. There was rain in the morning. Saifur though that he better stay at home.\"\n",
    "    }\n",
    "}\n",
    "\n",
    "# data = {\n",
    "# \"inputs\": {\n",
    "#     \"question\": \"What is used for inference?\",\n",
    "#     \"context\": \"My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference.\"\n",
    "#     }\n",
    "# }\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12d36533",
   "metadata": {},
   "source": [
    "## Predictor:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4c29f6e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "preds = predictor.predict(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "54d8ddd2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'score': 0.16734573245048523, 'start': 57, 'end': 61, 'answer': 'rain'}"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preds\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "72a66518",
   "metadata": {},
   "outputs": [],
   "source": [
    "bucket = \"sagemaker-us-west-2-893195739154\"\n",
    "input_prefix = \"sagemaker/input\"\n",
    "output_prefix = \"sagemaker/output\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22d127f8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_python3",
   "language": "python",
   "name": "conda_python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
