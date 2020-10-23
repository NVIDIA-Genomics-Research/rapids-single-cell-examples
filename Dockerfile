ARG BASE_IMAGE=rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.8

FROM ${BASE_IMAGE}
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git python3-setuptools python3-pip build-essential

RUN /opt/conda/envs/rapids/bin/pip install \
    scanpy wget python-igraph louvain leidenalg scanpy

WORKDIR /workspace

RUN git clone \
    https://github.com/clara-parabricks/rapids-single-cell-examples.git \
    rapids-single-cell-examples
WORKDIR /workspace/rapids-single-cell-examples

ARG GIT_BRANCH=master
RUN git checkout ${GIT_BRANCH}

RUN mkdir -p /opt/nvidia/scrna/
COPY launch /opt/nvidia/scrna/

RUN /opt/conda/envs/rapids/bin/python3 /opt/nvidia/scrna/launch create_env -e hlca_lung
RUN /opt/conda/envs/rapids/bin/python3 /opt/nvidia/scrna/launch create_env -e dsci_bmmc_60k
RUN /opt/conda/envs/rapids/bin/python3 /opt/nvidia/scrna/launch create_env -e dsci_bmmc_60k_viz
RUN /opt/conda/envs/rapids/bin/python3 /opt/nvidia/scrna/launch create_env -e 1M_brain

CMD /opt/conda/envs/rapids/bin/jupyter-lab \
		--no-browser \
		--port=8888 \
		--ip=0.0.0.0 \
		--notebook-dir=/workspace \
		--NotebookApp.password=\"\" \
		--NotebookApp.token=\"\" \
		--NotebookApp.password_required=False
