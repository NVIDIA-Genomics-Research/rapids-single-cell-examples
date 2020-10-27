ARG BASE_IMAGE=rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.8

FROM ${BASE_IMAGE}
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git python3-setuptools python3-pip build-essential

RUN /opt/conda/envs/rapids/bin/pip install \
    scanpy python-igraph louvain leidenalg wget

WORKDIR /workspace
ENV HOME /workspace
RUN git clone \
    https://github.com/clara-parabricks/rapids-single-cell-examples.git \
    rapids-single-cell-examples

ARG GIT_BRANCH=master
RUN cd rapids-single-cell-examples && git checkout ${GIT_BRANCH} && git pull

RUN mkdir -p /opt/nvidia/scrna/
COPY launch /opt/nvidia/scrna/

SHELL ["/bin/bash", "-c"]
RUN cd /workspace/rapids-single-cell-examples && source activate rapids && \
	python3 /opt/nvidia/scrna/launch create_env -e dsci_bmmc_60k_viz  && \
	python3 /opt/nvidia/scrna/launch create_env -e hlca_lung && \
	python3 /opt/nvidia/scrna/launch create_env -e dsci_bmmc_60k  && \
	python3 /opt/nvidia/scrna/launch create_env -e 1M_brain

RUN source activate rapidgenomics && \
	python3 -m ipykernel install --user --name=rapidgenomics
RUN source activate rapidgenomics_viz && \
	python3 -m ipykernel install --user --name=rapidgenomics_viz

RUN /opt/conda/envs/rapidgenomics/bin/pip install wget
RUN /opt/conda/envs/rapidgenomics_viz/bin/pip install wget

CMD jupyter-lab \
		--no-browser \
		--port=8888 \
		--ip=0.0.0.0 \
		--notebook-dir=/workspace \
		--NotebookApp.password="" \
		--NotebookApp.token="" \
		--NotebookApp.password_required=False
