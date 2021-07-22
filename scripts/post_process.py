import sys

if __name__ == "__main__":
    for line in sys.stdin:
        idx, _, sent = line.strip().split('\t')
        sent = " ".join(sent.split(' ')[1:])
        sys.stdout.write("\t".join([idx.split('-')[-1], sent])+'\n')