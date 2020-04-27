#! /bin/env bash

tar_url=$1
download_dir=$2

#######################

# Download fastqs
cd $download_dir
wget $tar_url

# Extract fastqs
fastq_dir=$(basename $tar_url .tar)
tar -xvf $fastq_dir.tar
