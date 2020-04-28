#! /bin/env bash

fastq_dir=$1
barcode_fastq_pattern=$2
read_fastq_pattern=$3
out_dir=$4

# Combine barcode fastqs
cat $fastq_dir/$barcode_fastq_pattern > $out_dir/barcodes.fastq.gz

# Combine read fastqs
cat $fastq_dir/$read_fastq_pattern > $out_dir/reads.fastq.gz