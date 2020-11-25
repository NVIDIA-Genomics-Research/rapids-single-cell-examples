#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -e

BASE_DIR=$(dirname $0)
# IMAGE_NAME=claraparabricks/single-cell-examples_rapids_cuda11.0
IMAGE_NAME=claraparabricks/single-cell-examples_rapids_cuda10.2
# BASE_IMAGE=rapidsai/rapidsai:cuda11.0-runtime-ubuntu18.04-py3.8
# BASE_IMAGE=rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.8
BASE_IMAGE=rapidsai/rapidsai-dev:0.16-cuda10.2-devel-ubuntu18.04-py3.8
GIT_BRANCH='master'
CONTAINER_TAG='v0.0.1'

HELP=false
PUSH=false

buildContainer() {

  local image_name=$1
  local container_tag=$2
  local base_image=$3
  local git_branch=$4

  echo "Creating image ${image_name}:${container_tag} from ${base_image}..."
  docker build --network=host -t ${image_name}:${container_tag} \
    --build-arg BASE_IMAGE="${base_image}" \
    --build-arg GIT_BRANCH="${git_branch}" \
    ${BASE_DIR}
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  -b | --base-image)
    BASE_IMAGE=$2
    shift
    shift
    ;;
  -t | --tag)
    CONTAINER_TAG=$2
    shift
    shift
    ;;
  -g | --git-branch)
    GIT_BRANCH=$2
    shift
    shift
    ;;
  -p | --prod-push)
    PUSH=true
    shift
    ;;
  -h | --help)
    HELP=true
    shift
    ;;
  *)
    HELP=true
    echo "Unknown key found : $key"
    shift
    ;;
  esac
  if [ ${HELP} = true ]; then
    break
  fi
done


if [ ${HELP} = true ]; then
	cat <<EOF
Builds scRNA example container.

Options:
  -b --base-image: Base docker image. Please refer https://rapids.ai/
  -t --tag       : Image tag for the new image
  -g --git-branch: Repo version to be used from https://github.com/clara-parabricks/rapids-single-cell-examples
  -p --prod-push : Push the new image to repo
  -h --help      : Display this help/usage message

EOF
  exit 0
fi

buildContainer ${IMAGE_NAME} ${CONTAINER_TAG} ${BASE_IMAGE} ${GIT_BRANCH}

if [ ${PUSH} == true ]; then
  echo "Pushing the images to repository..."
  docker tag ${IMAGE_NAME}:${CONTAINER_TAG} ${IMAGE_NAME}:latest

  docker push ${IMAGE_NAME}:${CONTAINER_TAG}
  docker push ${IMAGE_NAME}:latest
fi
