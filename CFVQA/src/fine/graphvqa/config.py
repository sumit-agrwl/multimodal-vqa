from fine.constants import constants
import json

class GraphVQAConfig(object):
    def __init__(self,
                vocab_size,
                sg_vocab_size,
                text_emb_dim=300, #glove embedding dimension
                max_execution_step=5,
                question_hidden_dim=512,
                init_token = 2,
                eos_token = 3,
                unk_token = 0,
                pad_token = 1,
                sg_init_token = 2,
                sg_eos_token = 3,
                sg_unk_token = 0,
                sg_pad_token = 1,
                no_encoder_attn_heads = 8,
                no_program_decoder_attn_heads = 8,
                no_full_answer_decoder_attn_heads = 8,
                no_gat_attn_heads = 4,
                dropout = 0.1,
                no_encoder_layers = 3,
                no_program_decoder_layers = 3,
                no_full_answer_decoder_layers = 3):
        self.vocab_size = vocab_size
        self.max_execution_step = max_execution_step
        self.text_emb_dim = text_emb_dim
        self.question_hidden_dim = question_hidden_dim
        self.vocab_size = vocab_size
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sg_vocab_size = sg_vocab_size
        self.sg_init_token = sg_init_token
        self.sg_eos_token = sg_eos_token
        self.sg_unk_token = sg_unk_token
        self.sg_pad_token = sg_pad_token
        self.no_encoder_attn_heads = no_encoder_attn_heads
        self.no_program_decoder_attn_heads = no_program_decoder_attn_heads
        self.no_full_answer_decoder_attn_heads = no_full_answer_decoder_attn_heads
        self.no_gat_attn_heads = no_gat_attn_heads
        self.dropout = dropout
        self.no_encoder_layers = no_encoder_layers
        self.no_program_decoder_layers = no_program_decoder_layers
        self.no_full_answer_decoder_layers = no_full_answer_decoder_layers

with open(constants.CONFIG_FILE) as f:
    config = json.load(f)
graphVQAConfig = GraphVQAConfig(**config)