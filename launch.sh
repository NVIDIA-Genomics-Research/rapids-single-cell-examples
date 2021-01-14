#!/bin/bash

LOCAL_ENV=.local_env
CREATE_ENV=false

usage() {
	cat <<EOF
USAGE: launch.sh

launch utility script
----------------------------------------

launch.sh [command]
	build
	bash
	setup
	container
	jupyter
EOF
	exit
}


if [ -e ./$LOCAL_ENV ]
then
	echo sourcing environment from ./$LOCAL_ENV
	. ./$LOCAL_ENV
else
	echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
	CREATE_ENV=true
fi

CONT='claraparabricks/single-cell-examples_rapids_cuda11.0:v0.0.1'
JUPYTER_PORT=${JUPYTER_PORT:-8888}
PLOTLY_PORT=${PLOTLY_PORT:-5000}
DASK_PORT=${DASK_PORT:-9001}
PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
DATA_PATH=${DATA_PATH:=../data}

if [ ${CREATE_ENV} = true ]; then
	echo CONT=${CONT} >> $LOCAL_ENV
	echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
	echo PLOTLY_PORT=${PLOTLY_PORT} >> $LOCAL_ENV
	echo DASK_PORT=${DASK_PORT} >> $LOCAL_ENV
	echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
	echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
fi

DOCKER_VERSION_WITH_GPU_SUPPORT="19.03.0"
DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2'})

PARAM_RUNTIME="--runtime=nvidia"
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ];
then
    PARAM_RUNTIME="--gpus all"
fi

DOCKER_CMD="docker run \
		${PARAM_RUNTIME} \
        --network host \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-e HOME=/workspace \
		-e TF_CPP_MIN_LOG_LEVEL=3 \
		-p ${JUPYTER_PORT}:8888 \
		-p ${DASK_PORT}:8787 \
		-p ${PLOTLY_PORT}:5000 \
		-v ${DATA_PATH}:/workspace/rapids-single-cell-examples/data \
		-v ${PROJECT_PATH}:/workspace/examples \
		-w /workspace/examples"

JUPYTER_CMD="/opt/conda/envs/rapids/bin/jupyter-lab \
		--no-browser \
		--allow-root \
		--port=8888 \
	    --allow-root \
		--ip=0.0.0.0 \
		--notebook-dir=/workspace/examples \
		--NotebookApp.password=\"\" \
		--NotebookApp.token=\"\" \
		--NotebookApp.password_required=False"

build() {
	echo 'Building container...'
	./build.sh
	exit
}


bash() {
	${DOCKER_CMD} -it ${CONT} bash
	exit
}


setup() {
	local DATA_DIR=${DATA_PATH}/data

	if [ ! -d "$DATA_DIR" ]; then
		echo "Downloading datasets..."
		mkdir -p ${DATA_DIR}

		echo 'Downloading 70k krasnow_hlca_10x dataset...'
		wget -q --show-progress \
			-p ${DATA_DIR} \
			https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/krasnow_hlca_10x.sparse.h5ad

		echo 'Downloading dsci dataset...'
		wget -q --show-progress \
			-p ${DATA_DIR} \
			https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_nonzeropeaks.h5ad; \

		wget -q --show-progress \
			-p ${DATA_DIR} \
			https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_peaknames_nonzero.npy; \

		wget -q --show-progress \
			-p ${DATA_DIR} \
			https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/dsci_resting_cell_metadata.csv

		echo 'Downloading 1M brain_cells_10X dataset...'
		wget -q --show-progress \
			-p ${DATA_DIR} \
			https://rapids-single-cell-examples.s3.us-east-2.amazonaws.com/1M_brain_cells_10X.sparse.h5ad
	fi
}

container() {
	${DOCKER_CMD} -it ${CONT} ${JUPYTER_CMD}
	exit
}


jupyter() {
	${JUPYTER_CMD} --allow-root
	exit
}


case $1 in
	build)
		;&
	bash)
		;&
	setup)
		;&
	container)
		$1
		;;
	jupyter)
		$1
		;;
	*)
	usage
		;;
esac
