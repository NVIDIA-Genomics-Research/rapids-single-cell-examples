import sys
import argparse
import operator
import itertools
import traceback
import os.path
import multiprocessing
import pysam
import HTSeq
import time

# Pass if encountering an unknown chromosome
class UnknownChrom(Exception):
    pass

# Check read and mapping quality
def check_read(r, minaqual):
    #start = time.time()
    ret = r.aligned and (not r.not_primary_alignment) and (not r.supplementary) and (r.aQual >= minaqual) and r.optional_field("NH") == 1
    #print("check_read: " + str(time.time() - start))
    return ret


# Find genomic features that overlap with read
def get_overlapping_features(r, features):
    start = time.time()
    iv_seq = (co.ref_iv for co in r.cigar if co.type in com and co.size > 0)
    fs = set()
    for iv in iv_seq:
        if iv.chrom not in features.chrom_vectors:
            raise UnknownChrom
        for iv2, fs2 in features[iv].steps():
            fs = fs.union(fs2)
    ret = fs
    return ret


# Write output in tsv form
def write_to_out(r, assignment, outfile):
    name = r.read.name
    outfile.write(name + "\t" + assignment + "\n")


# Get features from GFF
def GFFToFeatures(gff_filename, stranded):
    start = time.time()
    gff = HTSeq.GFF_Reader(gff_filename)
    feature_scan = HTSeq.make_feature_genomicarrayofsets(
        gff,
        id_attribute='gene_name',
        feature_type='exon',
        feature_query=None,
        stranded=stranded != 'no',
        verbose=True)
    features = feature_scan['features']
    ret = features
    print("Loaded features in: " + str(time.time() - start))
    return ret


# CIGAR match characters (including alignment match, sequence match, and sequence mismatch
com = ('M', '=', 'X')


def main():

    pa = argparse.ArgumentParser(
        usage="%(prog)s [options] alignment_file gff_file",
        description="Takes an alignment file in SAM/BAM format and a feature file in GFF format, anoverlapsd returns .")

    pa.add_argument("sam_file", type=str, help="Path to the SAM/BAM file containing the mapped reads.")
    pa.add_argument("features_file", type=str, help="Path to the GFF file containing the features")
    pa.add_argument("-s", "--stranded", dest="stranded", choices=("yes", "no"), default="yes",
            help="Whether the data is strand-specific. 'yes', or 'no'")
    pa.add_argument("-a", "--minaqual", type=int, dest="minaqual",
            default=10, help="Skip all reads with MAPQ alignment quality lower than this.")
    pa.add_argument("-o", "--out", type=str, dest="out",
            action='append', help="Write overlaps to this file")

    args = pa.parse_args()

    # Prepare features
    print("Loading features")
    features = GFFToFeatures(args.features_file, args.stranded)

    # Open input file
    print("Loading input file")
    read_seq_iter = iter(HTSeq.BAM_Reader(args.sam_file))

    # Initialize counts
    i = 0

    # Open output file
    print("Opening output file handle")
    #outfile = open(args.out[0], 'w')

    print("Iterating over reads")
    for read in read_seq_iter:

        # Track number of reads
        if i > 0 and i % 1000000 == 0:
            sys.stderr.write("%d alignment records processed.\n" %i)
            sys.stderr.flush()
        i += 1

        # If read is good, get aligned genomic intervals
        if check_read(read, args.minaqual):            

            
            # Overlap the read-aligned genomic intervals with features.
            fs = get_overlapping_features(read, features)

            # Write output if read overlaps with one feature only.
            if fs is not None and len(fs) == 1:
                pass
    #            write_to_out(read, list(fs)[0], outfile)

    #outfile.close()


if __name__ == "__main__":
    main()
