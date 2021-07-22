from fairseq.models import (
    register_model,
    register_model_architecture,
)


# New: model parallel transformer
@register_model_architecture("model_parallel_transformer", "transformer_mp")
def base_architecture_mp(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)


@register_model_architecture("model_parallel_transformer", "transformer_vaswani_wmt_en_de_big_mp")
def transformer_vaswani_wmt_en_de_big_mp(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture_mp(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("model_parallel_transformer", "transformer_wmt_en_de_big_t2t_mp")
def transformer_wmt_en_de_big_t2t_mp(args):
    # use pre-norm
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big_mp(args)


@register_model_architecture('model_parallel_transformer', 'transformer_prenorm_6e6d_emb_1024_mp')
def transformer_prenorm_6e6d_emb_1024_mp(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    transformer_wmt_en_de_big_t2t_mp(args)


@register_model_architecture('model_parallel_transformer', 'transformer_prenorm_12e12d_emb_2048_mp')
def transformer_prenorm_12e12d_emb_2048_mp(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 2048)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    transformer_wmt_en_de_big_t2t_mp(args)


@register_model_architecture('model_parallel_transformer', 'transformer_prenorm_24e24d_emb_1024_mp')
def transformer_prenorm_24e24d_emb_1024_mp(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    transformer_wmt_en_de_big_t2t_mp(args)


@register_model_architecture('model_parallel_transformer', 'transformer_prenorm_24e24d_emb_4096_mp')
def transformer_prenorm_24e24d_emb_4096_mp(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 4096)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    transformer_wmt_en_de_big_t2t_mp(args)
