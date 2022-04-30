FROM ubuntu:18.04
FROM python:3.9.7-slim  

#Server components to support multiple parallel incoming requests
FROM nginx:1.10.1-alpine
COPY ./nginx.conf /etc/nginx/nginx.conf

LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies for SageMaker Inference Toolkit

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Install MXNet, MMS, and SageMaker Inference Toolkit to set up MMS
RUN pip3 --no-cache-dir install mxnet \
                                multi-model-server \
                                sagemaker-inference \
                                retrying
# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Copy entrypoint script to the image
COPY dockerd-entrypoint.py /usr/local/bin/dockerd-entrypoint.py
RUN chmod +x /usr/local/bin/dockerd-entrypoint.py

RUN mkdir -p /home/model-server/

# Copy the default custom service file to handle incoming data and inference requests
COPY model_handler.py /home/model-server/model_handler.py

# Define an entrypoint script for the docker image
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]

# Define command to be passed to the entrypoint
CMD ["serve"]


# Hugging-face Model Specific
RUN curl https://aws-dlc-licenses.s3.amazonaws.com/pytorch-1.9/license.txt -o /license.txt

# Uninstall and re-install torch and torchvision from the PyTorch website
RUN pip uninstall -y torch \
 && pip install --no-cache-dir -U $PT_INFERENCE_URL

# install Hugging Face libraries and its dependencies
RUN pip install --no-cache-dir \
	transformers[sentencepiece]==${TRANSFORMERS_VERSION} \
	protobuf==3.12.0 \
	"sagemaker-huggingface-inference-toolkit<2"

RUN HOME_DIR=/root \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
 && rm -rf ${HOME_DIR}/oss_compliance*

EXPOSE 8080 8081
ENTRYPOINT ["python", "/usr/local/bin/dockerd-entrypoint.py"]
CMD ["serve"]