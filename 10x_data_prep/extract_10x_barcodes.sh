#! /bin/env bash

bc_pattern=$1
barcode_fastq=$2
read_fastq=$3
cells=$4
out_dir=$5

# Identify correct cell barcodes
umi_tools whitelist --stdin $barcode_fastq --bc-pattern $bc_pattern --set-cell-number $cells --log2stderr > $out_dir/whitelist.txt
                    
# Extract barcdoes and UMIs and add to read names
umi_tools extract --bc-pattern $bc_pattern --stdin $barcode_fastq --stdout $out_dir/barcodes_extracted.fastq.gz --read2-in $read_fastq --read2-out $out_dir/reads_extracted.fastq.gz --filter-cell-barcode --whitelist $out_dir/whitelist.txt