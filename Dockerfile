ARG BASE_IMAGE=rapidsai/rapidsai:21.08-cuda11.2-runtime-ubuntu18.04-py3.7

FROM ${BASE_IMAGE}
RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git python3-setuptools python3-pip build-essential libcurl4-gnutls-dev \
    zlib1g-dev rsync vim cmake tabix

RUN /opt/conda/bin/conda install -y \
    scanpy==1.8.1 dash-html-components dash-bootstrap-components dash-core-components dash-daq

RUN /opt/conda/envs/rapids/bin/pip install \
    wget pytabix atacworks==0.3.4 
    
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

