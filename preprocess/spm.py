import sentencepiece as spm
import sys

if __name__ == "__main__":
    spm_encoder = spm.SentencePieceProcessor(model_file=sys.argv[1])
    for line in sys.stdin:
        tokens = spm_encoder.encode(line.strip(), out_type=str)
        spm_text = " ".join(tokens)
        sys.stdout.write(spm_text+'\n')