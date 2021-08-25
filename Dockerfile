ARG BASE_IMAGE=rapidsai/rapidsai:21.08-cuda11.0-runtime-ubuntu18.04-py3.7

FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git python3-setuptools python3-pip build-essential libcurl4-gnutls-dev \
    zlib1g-dev rsync vim cmake tabix

RUN /opt/conda/envs/rapids/bin/pip install \
    scanpy==1.7.2 wget pytabix dash-daq atacworks==0.3.4 \
    dash-html-components dash-bootstrap-components dash-core-components

RUN /opt/conda/envs/rapids/bin/pip install --ignore-installed numba==0.52.0

WORKDIR /workspace
ENV HOME /workspace
RUN git clone \
    https://github.com/clara-parabricks/rapids-single-cell-examples.git \
    rapids-single-cell-examples

ARG GIT_BRANCH=master
RUN cd rapids-single-cell-examples && git checkout ${GIT_BRANCH} && git pull

CMD jupyter-lab \
		--no-browser \
		--allow-root \
		--port=8888 \
		--ip=0.0.0.0 \
		--notebook-dir=/workspace \
		--NotebookApp.password="" \
		--NotebookApp.token="" \
		--NotebookApp.password_required=False

# ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/compat
# RUN echo "export PATH=$PATH:/workspace/data" >> ~/.bashrc
