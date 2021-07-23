import numpy as np
import torch
import argparse
import os
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.data import encoders
from tqdm import tqdm
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf

np.set_printoptions(precision=10)


parser = options.get_generation_parser()
parser.add_argument('--ckpt', type=str, metavar='N', help='checkpoint path')
parser.add_argument('--dest', type=str, metavar='N', help='destination')
parser.add_argument('--architect', type=str, metavar='N', help='architecture')
# options.add_dataset_args(parser)
args = options.parse_args_and_arch(parser)

print(args)
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)


def get_avg(tokens, lengths, model, has_langtok=False):
    encoder_outs = model.encoder.forward(tokens, lengths)
    np_encoder_outs = encoder_outs.encoder_out.detach().cpu().numpy().astype(np.float32)
    if encoder_outs.encoder_padding_mask is not None:
        encoder_mask = 1 - encoder_outs.encoder_padding_mask.detach().cpu().numpy().astype(np.float32)
        encoder_mask = np.expand_dims(encoder_mask.T, axis=2)
    else:
        encoder_mask = torch.ones(encoder_outs.encoder_out.shape)
    if has_langtok:
        encoder_mask = encoder_mask[1:, :, :]
        np_encoder_outs = np_encoder_outs[1, :, :]
    masked_encoder_outs = encoder_mask * np_encoder_outs
    avg_pool = (masked_encoder_outs / encoder_mask.sum(axis=0)).sum(axis=0)
    return avg_pool


def _get_avg(vector, padding_mask):
    np_vector = vector.detach().cpu().numpy().astype(np.float32)
    if padding_mask is not None:
        _mask = 1 - padding_mask.detach().cpu().numpy().astype(np.float32)
        _mask = np.expand_dims(_mask.T, axis=2)
    else:
        _mask = torch.ones(vector.shape)
    masked_np_vector = _mask * np_vector
    avg_pool = (masked_np_vector / _mask.sum(axis=0)).sum(axis=0)
    return avg_pool


def get_orth_avg(tokens, lengths, model):
    encoder_out, (lang_vec, semantic_vec) = model.encoder.forward_heads(tokens, lengths)
    lang_emb = _get_avg(lang_vec, encoder_out.encoder_padding_mask)
    semantic_emb = _get_avg(semantic_vec, encoder_out.encoder_padding_mask)
    return lang_emb, semantic_emb


def get_hidden_states(ckpt):
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt)
    src_dict = getattr(task, 'source_dictionary', None)
    tgt_dict = task.target_dictionary
    
    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    subword = encoders.build_bpe(args)
    
    # if isinstance(cfg, argparse.Namespace):
    #     cfg = convert_namespace_to_omegaconf(cfg)
    
    # model = task.build_model(saved_args)
    model = task.build_model(state["args"])
    model.load_state_dict(state["model"], strict=True)
    model.cuda()
    
    def decode_fn(x):
        if subword is not None:
            x = subword.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        x = "".join(x.split(" ")).replace("▁", " ").strip()
        return x
    
    def toks_2_sent(toks):
        _str = tgt_dict.string(toks)  # , args.remove_bpe
        _sent = decode_fn(_str)
        return _sent
    
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions()
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        # data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    
    # initialize
    src_sentences = []
    src_hidden_states_list = []
    idx_list = []
    
    for sample in tqdm(itr):
        sample = utils.move_to_cuda(sample)
        if args.architect == "orthogonal_head_transformer_t2t_12e12d":
            _, src_avg_states = get_orth_avg(sample["net_input"]["src_tokens"],
                                             sample["net_input"]["src_lengths"], model)
        else:
            src_avg_states = get_avg(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"], model, False)
        src_hidden_states_list.extend(src_avg_states)
        idx_list.extend(sample["id"].detach().cpu().numpy())
        for i, sample_id in enumerate(sample['id'].tolist()):
            src_tokens_i = utils.strip_pad(sample['net_input']['src_tokens'][i, :], src_dict.pad())
            src_sent_i = toks_2_sent(src_tokens_i)
            src_sentences.append(src_sent_i)
    return src_sentences, src_hidden_states_list, idx_list


source = args.source_lang.split("_")[0]
print("=====Start Loading=====")
ckpt_path = args.ckpt

dest = args.dest if args.dest else "."

os.system("mkdir -p {}".format(dest))
src_sents, src_values, indexes = get_hidden_states(ckpt_path)
print("=====End Loading=====")
with open("{}/sentences.{}".format(dest, source), "w") as fwe:
    for _id, src_w in enumerate(src_sents):
        fwe.write("{}\t{}\n".format(indexes[_id], src_w))

np.savetxt("{}/sent_avg_pool.{}".format(dest, source), src_values, delimiter=',')

print(len(src_values), len(src_values[0]))

