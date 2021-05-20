from fairseq.models import register_model_architecture


@register_model_architecture('transformer', 'transformer_bigger')
def transformer_bigger(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.3)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.3)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 15000)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 15000)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)
    
    
@register_model_architecture('transformer', 'transformer_bigger_16384')
def transformer_bigger_16384(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 16384)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 16384)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_bigger_no_share')
def transformer_bigger_no_share(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.3)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.3)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 15000)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 15000)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_deeper')
def transformer_deeper(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 15)
    args.dense = False
    args.bottleneck_component = getattr(args, 'bottleneck_component', 'mean_pool')
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 15000)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 15000)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_deeper_no_share')
def transformer_deeper_no_share(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 15)
    args.dense = False
    args.bottleneck_component = getattr(args, 'bottleneck_component', 'mean_pool')
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 15000)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 15000)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_deeper_dense')
def transformer_deeper_no_share(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 15)
    args.dense = True
    args.bottleneck_component = 'mean_pool'
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 15000)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 15000)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_deeper_dense_no_share')
def transformer_deeper_no_share(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 15)
    args.dense = True
    args.bottleneck_component = 'mean_pool'
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.dropout = getattr(args, 'dropout', 0.1)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 15000)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 15000)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_big')
def transformer_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_big_emb512')
def transformer_big_emb512(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_big_no_share')
def transformer_big_no_share(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_big_16e4d')
def transformer_big_16e4d(args):
    args.dropout = getattr(args, 'dropout', 0.2)
    args.encoder_layers = getattr(args, 'encoder_layers', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_big_16e6d')
def transformer_big_16e6d(args):
    args.dropout = getattr(args, 'dropout', 0.2)
    args.encoder_layers = getattr(args, 'encoder_layers', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_base')
def transformer_bigger(args):
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de
    transformer_wmt_en_de(args)


@register_model_architecture('transformer', 'transformer_mid_50e6d')
def transformer_mid_50e6d(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 50)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_big_t2t_12e12d')
def transformer_big_t2t_12e12d(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'mix_transformer_mid_50e6d')
def mix_transformer_mid_50e6d(args):
    args.mix_prepost_norm  = getattr(args, "mix_prepost_norm", True)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 50)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.mix_type = getattr(args, "mix_type", "learnable")
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 're_zero_transformer_mid_50e6d')
def re_zero_transformer_mid_50e6d(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 50)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.re_zero = getattr(args, "re_zero", True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)
    

@register_model_architecture('transformer', 'transformer_mid_50e3d_ed3072')
def transformer_mid_50e3d_ed3072(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 50)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'mix_transformer_mid_50e6d_3000fix_10000decay')
def mix_transformer_mid_50e6d_3000fix_10000decay(args):
    args.mix_prepost_norm  = getattr(args, "mix_prepost_norm", True)
    args.mix_type = getattr(args, "mix_type", "step_moving")
    args.pre_steps = getattr(args, "pre_steps", 3000)
    args.change_steps = getattr(args, "change_steps", 10000)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 50)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'mix_transformer_mid_50e6d_7000fix_7000decay')
def mix_transformer_mid_50e6d_3000fix_10000decay(args):
    args.mix_prepost_norm = getattr(args, "mix_prepost_norm", True)
    args.mix_type = getattr(args, "mix_type", "step_moving")
    args.pre_steps = getattr(args, "pre_steps", 7000)
    args.change_steps = getattr(args, "change_steps", 7000)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 50)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_mid_75e6d')
def transformer_mid_75e6d(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 75)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_mid_25e6d')
def transformer_mid_25e6d(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 25)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)
    

@register_model_architecture('transformer', 'transformer_mid_25e6d_ed3072')
def transformer_mid_25e6d_ed3072(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 25)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_mid_25e6d_e3072_d4096')
def transformer_mid_25e6d_e3072_d4096(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 25)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    # args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


# def transformer_fixed_multihead(args):
#     args.head_dim = getattr(args, 'head_dim', 128)
#     from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
#     transformer_wmt_en_de_big_t2t(args)

@register_model_architecture('transformer', 'transformer_fixed_multihead_base')
def transformer_fixed_multihead_base(args):
    args.head_dim = getattr(args, 'head_dim', 128)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_fixed_multihead_embed_1024_nhead_16_hdim_128')
def transformer_fixed_multihead_embed_1024_nhead_16_hdim_128(args):
    args.head_dim = getattr(args, 'head_dim', 128)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_fixed_multihead_embed_1024_nhead_16_hdim_256')
def transformer_fixed_multihead_embed_1024_nhead_16_hdim_128(args):
    args.head_dim = getattr(args, 'head_dim', 256)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_fh_16x128_layer_12')
def transformer_fh_16x128_layer_12(args):
    args.head_dim = getattr(args, 'head_dim', 128)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)


@register_model_architecture('transformer', 'transformer_fh_16x256_layer_12')
def transformer_fh_16x256_layer_12(args):
    args.head_dim = getattr(args, 'head_dim', 256)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    from fairseq.models.transformer import transformer_wmt_en_de_big_t2t
    transformer_wmt_en_de_big_t2t(args)
