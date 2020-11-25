ARG BASE_IMAGE=rapidsai/rapidsai:0.16-cuda11.0-runtime-ubuntu18.04-py3.8

FROM ${BASE_IMAGE}
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git python3-setuptools python3-pip build-essential vim

RUN /opt/conda/envs/rapids/bin/pip install \
    scanpy wget python-igraph louvain leidenalg

WORKDIR /workspace

RUN git clone \
    https://github.com/clara-parabricks/rapids-single-cell-examples.git \
    rapids-single-cell-examples
WORKDIR /workspace/rapids-single-cell-examples

ARG GIT_BRANCH=master
RUN git checkout ${GIT_BRANCH}

RUN mkdir -p /opt/nvidia/scrna/
COPY launch.sh /opt/nvidia/scrna/

CMD /opt/nvidia/scrna/launch.sh jupyter
