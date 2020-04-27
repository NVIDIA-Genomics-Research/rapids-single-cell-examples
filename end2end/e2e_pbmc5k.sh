# End to end pipeline with 10X PBMC 5k dataset

#! /bin/env bash

# Set variables

# Fastq file download link
tar_url=http://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/3.0.2/5k_pbmc_v3/5k_pbmc_v3_fastqs.tar

# Barcode pattern
bc_pattern=CCCCCCCCCCCCCCCCNNNNNNNNNNNN

# File names
barcode_fastq_pattern=5k_pbmc_v3_S1_L00?_R1_001.fastq.gz
read_fastq_pattern=5k_pbmc_v3_S1_L00?_R2_001.fastq.gz

# Number of cells to select
cells=5025

# GTF file
gtf=/covid-omics/scrna_demo/genome_files/gencode.v33.primary_assembly.annotation.gtf

# Genome index for STAR
genome_dir=/covid-omics/scrna_demo/genome_index/

# Directories
scripts=/covid-omics/scrna_demo/scripts
download_dir=/covid-omics/scrna_demo/data/5k_pbmcs

# Number of threads
threads=64

#######################

# No need to time these steps

#######################

# Create STAR index
#STAR --runThreadN $threads --runMode genomeGenerate --genomeDir $genome_dir --genomeFastaFiles /covid-omics/scrna_demo/genome_files/GRCh38.primary_assembly.genome.fa --sjdbGTFfile $gtf

fastq_dir=$(basename $tar_url .tar)

# Download and extract fastqs
bash $scripts/download_10x_fastqs.sh $tar_url $download_dir

# Combine fastqs
bash $scripts/combine_10x_fastqs.sh $download_dir/$fastq_dir $barcode_fastq_pattern $read_fastq_pattern $download_dir

# Extract read, barcode, and UMI sequences
bash $scripts/extract_10x_barcodes.sh $bc_pattern $download_dir/barcodes.fastq.gz $download_dir/reads.fastq.gz $cells $download_dir

# Count reads
#zcat $download_dir/reads_extracted.fastq.gz | grep "@"|wc -l
# 348847429

#######################

# Time steps from here

#######################

cd $download_dir

# Map reads
time STAR --runThreadN $threads --genomeDir $genome_dir --readFilesIn reads_extracted.fastq.gz --readFilesCommand zcat --outFilterMultimapNmax 1 --outSAMtype BAM SortedByCoordinate
#real	53m18.247s
#user	271m37.329s
#sys	12m11.262s

# Assign reads to genes
time featureCounts -a $gtf -o gene_assigned -R BAM Aligned.sortedByCoord.out.bam -T $threads -g gene_name 
#real	9m9.229s
#user	27m2.212s
#sys	1m14.780s

# Sort output BAM file 
time samtools sort -@ $threads Aligned.sortedByCoord.out.bam.featureCounts.bam -o assigned_sorted.bam
#real	19m58.209s
#user	66m22.017s
#sys	8m19.184s

# index output BAM file 
time samtools index assigned_sorted.bam
#real	5m49.702s
#user	4m12.661s
#sys	0m2.637s

# Count UMIs per gene per cell
time umi_tools count --per-gene --gene-tag=XT --assigned-status-tag=XS --per-cell -I assigned_sorted.bam -S counts.tsv --wide-format-cell-counts
#real 49m54.002s
#user 49m13.913s
#sys 0m35.987s
########################

# Normalize count matrix
time python $scripts/scanpy_preprocess_1_cpu.py --input counts.tsv --out_dir ./ --out_prefix 5k_pbmc --min_genes 200 --max_genes 6000 --max_mito 0.25
#Count matrix loading time: 19.228405131027102
#AnnData object with n_obs × n_vars = 5021 × 35085 
#AnnData object with n_obs × n_vars = 4999 × 35085 
#AnnData object with n_obs × n_vars = 4993 × 35085 
#View of AnnData object with n_obs × n_vars = 4648 × 35085 
#cell filtering time: 2.6173269720748067
#normalization time: 1.9090962633490562
#log transform time: 1.8341417033225298
#write time: 23.958261010237038
#real	0m51.280s
#user	0m27.863s
#sys	0m10.116s

# Filter HVGs
time python $scripts/scanpy_preprocess_2_cpu.py --input 5k_pbmc_scanpy_normalized_counts.h5ad --out_dir ./ --out_prefix 5k_pbmc --min_disp 0.2
#Selected 7282 genes.
#identifying HVG time: 3.04471190366894
#regression (on filtered data) time: 121.20122353639454
#scaling filtered data time: 0.3984785694628954
#write time: 7.360320990905166
#real	2m13.991s
#user	7m35.165s
#sys	79m23.094s

# Cluster
time python $scripts/scanpy_cluster_cpu.py --input 5k_pbmc_scanpy_scaled.h5ad --out_dir ./ --out_prefix 5k_pbmc --ncomps 50 --ncomps-knn 50 --neighbors 15 --resolution 0.5
