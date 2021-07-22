import json
import os
import argparse
import sentencepiece as spm
from collections import OrderedDict
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--langs', required=False, help="The iso 639-2 code of languages that we apply RAS.")
parser.add_argument('--spm-model', type=str, required=True)
parser.add_argument('--dict-path', type=str, required=True)
parser.add_argument('--vocab-size', type=int, default=100000000)
args = parser.parse_args()
s = spm.SentencePieceProcessor(model_file=args.spm_model)


# read dictionaries
def load_multi_dict(dictionary_path, languages=None):
    """

    :param dictionary_path: the path the multi-way dictionary
    :param languages: languages that will be loaded to the final dict
    :return: a dictionary of dictionaries that stores all synonyms in `languages`
    """
    word_dict = {}
    with open(os.path.join(dictionary_path)) as f:
        i = 0
        for _line in tqdm(f):
            _line = _line.strip()
            source = _line.split("\t")[0]
            src_lang = source[:2].lower()
            src_word = source[4:]
            src_ids = "-".join([str(id+1) for id in s.encode(src_word, out_type=int)])
            if languages is not None and src_lang not in languages:
                continue  # skip languages that are not in `languages`
            if src_lang not in word_dict:
                word_dict[src_lang] = dict()
            if src_ids not in word_dict[src_lang]:
                word_dict[src_lang][src_ids] = OrderedDict()
            for word_str in _line.split("\t")[1:]:
                lang = word_str[:2].lower()
                word = word_str[4:-3]
                word_ids = "-".join([str(id+1) for id in s.encode(word, out_type=int)])
                if word_ids == src_ids:
                    continue  # skip
                depth = word_str[-1]
                depth = int(depth)
                if depth not in word_dict[src_lang][src_ids]:
                    word_dict[src_lang][src_ids][depth] = []
                word_dict[src_lang][src_ids][depth].append((word_ids, lang))
            if i >= args.vocab_size:
                # only keep first `vocab_size` word pairs
                break
            i += 1
    return word_dict


if __name__ == "__main__":
    dicts = load_multi_dict(args.dict_path)
    with open("id_dict_0.json", 'w') as fw:
        json.dump(dicts, fw, indent=2)
