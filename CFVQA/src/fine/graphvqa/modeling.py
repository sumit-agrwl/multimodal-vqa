"""
This file defines the whole pipeline model (all neural modules).

TO DEBUG:
python pipeline_model.py
"""
import torch
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean, scatter_add
import logging
import torch_geometric
from fine.graphvqa.graph_layernorm import GraphLayerNorm
from fine.graphvqa.gat_skip import gat_seq # use second version of gat
from fine.graphvqa.config import graphVQAConfig
import math

from fine.vocab.vocab_utils import SGVocab, QAVocab

"""
Scene Graph Encoding Module For Ground Truth (Graph Neural Module)
Functional definition of scene graph encoding layer
Return: a callable operator, which is an initialized torch_geometric.nn graph neural layer
"""
def get_gt_scene_graph_encoding_layer(num_node_features, num_edge_features):

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super(EdgeModel, self).__init__()
            self.edge_mlp = Seq(
                Lin(2 * num_node_features + num_edge_features, num_edge_features),
                ReLU(),
                Lin(num_edge_features, num_edge_features)
                )

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest, edge_attr], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super(NodeModel, self).__init__()
            self.node_mlp_1 = Seq(
                Lin(num_node_features + num_edge_features, num_node_features),
                ReLU(),
                Lin(num_node_features, num_node_features)
                )
            self.node_mlp_2 = Seq(
                Lin(2 * num_node_features, num_node_features),
                ReLU(),
                Lin(num_node_features, num_node_features)
                )

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out], dim=1)
            return self.node_mlp_2(out)

    op = torch_geometric.nn.MetaLayer(EdgeModel(), NodeModel())
    return op


"""
Final Layer of Graph Execution Module
"""

class ConditionalGlobalAttention(torch.nn.Module):
    r"""Language-Conditioned Global soft attention layer

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( u[batch] ) \dot h_{\mathbf{\Theta}} ( \mathbf{x}_n ) \right)
        \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),
    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)

    """
    def __init__(self, num_node_features, num_out_features):
        super(ConditionalGlobalAttention, self).__init__()
        channels = num_out_features
        self.gate_nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, 1))
        self.node_nn = Seq(Lin(num_node_features, channels), ReLU(), Lin(channels, channels))
        self.ques_nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))
        # self.gate_nn = Lin(channels, 1)
        # self.node_nn = Lin(channels, channels)
        # self.nn = Lin(num_node_features, channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.gate_nn)
        torch_geometric.nn.inits.reset(self.node_nn)
        torch_geometric.nn.inits.reset(self.ques_nn)

    def forward(self, x, u, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        # gate = self.gate_nn(x).view(-1, 1)

        ##################################
        # Batch
        # shape: x - [ Num of Nodes, num_node_features] --> [ Num of Nodes, Feature Channels ]
        # shape: u - [ Batch Size, Feature Channels]
        # shape: u[batch] - [ Num of Nodes, Feature Channels]
        ##################################
        x = self.node_nn(x) # if self.node_nn is not None else x
        # print("x", x.size(), "u", u.size(), "u[batch]", u[batch].size())

        ##################################
        # torch.bmm
        # batch1 and batch2 must be 3D Tensors each containing the same number of matrices.
        # If batch1 is a b x n x m Tensor, batch2 is a b x m x p Tensor, out will be a b x n x p Tensor.
        ##################################


        gate = self.gate_nn(self.ques_nn(u)[batch] * x)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        # gate = torch.bmm(x.unsqueeze(1) , self.ques_nn(u)[batch].unsqueeze(2)).squeeze(-1)
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out

    def __repr__(self):
        return '{}(gate_nn={}, node_nn={}, ques_nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.node_nn, self.ques_nn)

"""
Transformer for text
"""

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerProgramDecoder(torch.nn.Module):
    # should also be hierarchical

    def __init__(self, text_vocab_embedding, vocab_size, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerProgramDecoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        ##################################
        # For Hierarchical Deocding
        ##################################
        self.num_queries = graphVQAConfig.max_execution_step
        self.query_embed = torch.nn.Embedding(self.num_queries, ninp)

        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.coarse_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))

        ##################################
        # Decoding
        ##################################
        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        self.ninp = ninp

        self.vocab_decoder = torch.nn.Linear(ninp, vocab_size)


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, tgt):

        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=memory, tgt_mask=None) # [ MaxNumSteps, Batch, Dim]
        instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]

        ##################################
        # prepare target mask
        ##################################
        n_len_seq = tgt.shape[0] # seq len
        tgt_mask = self.generate_square_subsequent_mask(
                n_len_seq).to(memory.device)

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        tgt   = self.text_vocab_embedding(tgt)
        tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)

        ##################################
        # Replace the init token feature with instruciton feature
        ##################################

        tgt = tgt[1:] # [Len, Batch, Dim] discard the start of sentence token
        tgt = torch.cat((instr_vectors_reshape, tgt), dim=0) # replace with our init values

        output = self.transformer_decoder(tgt=tgt, memory=memory_repeat, tgt_mask=tgt_mask)
        output = self.vocab_decoder(output)

        # output both prediction and instruction vectors
        return output, instr_vectors

    def sample(self, memory, tgt):

        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(1, true_batch_size, 1) # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(tgt=instr_queries, memory=memory, tgt_mask=None) # [ MaxNumSteps, Batch, Dim]
        instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]


        tgt = None # discard

        max_output_len = 16 # 80 # program concat 80, full answer max 15, instr max 10
        batch_size = memory.size(1) * self.num_queries

        output = torch.ones(max_output_len, batch_size).long().to(memory.device) * graphVQAConfig.init_token


        for t in range(1, max_output_len):
            tgt = self.text_vocab_embedding(output[:t,:]) # from 0 to t-1
            tgt = self.emb_proj(tgt) * math.sqrt(self.ninp)
            tgt = self.pos_encoder(tgt) # contains dropout

            ##################################
            # Replace the init token feature with instruciton feature
            ##################################
            tgt = tgt[1:] # [Len, Batch, Dim] discard the start of sentence token
            tgt = torch.cat((instr_vectors_reshape, tgt), dim=0) # replace with our init values

            n_len_seq = t # seq len
            tgt_mask = self.generate_square_subsequent_mask(
                    n_len_seq).to(memory.device)
            # 2D mask (query L, key S)(L,S) where L is the target sequence length, S is the source sequence length.
            out = self.transformer_decoder(tgt, memory_repeat, tgt_mask=tgt_mask)
            # output: (T, N, E): target len, batch size, embedding size
            out = self.vocab_decoder(out)
            # target len, batch size, vocab size
            output_t = out[-1, :, :].data.topk(1)[1].squeeze()
            output[t,:] = output_t

        return output, instr_vectors

class TransformerQuestionEncoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerQuestionEncoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp) )
        self.ninp = ninp

    def forward(self, src):

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        src   = self.text_vocab_embedding(src)
        src = self.emb_proj(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TransformerInstrEncoder(torch.nn.Module):

    def __init__(self, text_vocab_embedding, text_emb_dim, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerInstrEncoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = 'Transformer'
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp) )
        self.ninp = ninp

    def forward(self, src):

        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        src   = self.text_vocab_embedding(src)
        src = self.emb_proj(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class SGEncoder(torch.nn.Module):
    def __init__(self):
        super(SGEncoder, self).__init__()
        self.sg_emb_dim = graphVQAConfig.text_emb_dim # 300d glove
        sg_pad_idx = graphVQAConfig.sg_pad_token
        self.sg_vocab_embedding = torch.nn.Embedding(graphVQAConfig.sg_vocab_size, self.sg_emb_dim, padding_idx=sg_pad_idx)
        # self.sg_vocab_embedding.weight.data.copy_(sg_vocab.vectors)

        ##################################
        # build scene graph encoding layer
        ##################################
        self.scene_graph_encoding_layer = get_gt_scene_graph_encoding_layer(
            num_node_features=self.sg_emb_dim,
            num_edge_features=self.sg_emb_dim)

        self.graph_layer_norm = GraphLayerNorm(self.sg_emb_dim)

    def forward(self, gt_scene_graphs):

        ##################################
        # Use glove embedding to embed ground truth scene graph
        ##################################
        # [ num_nodes, MAX_OBJ_TOKEN_LEN] -> [ num_nodes, MAX_OBJ_TOKEN_LEN, sg_emb_dim]
        x_embed = self.sg_vocab_embedding(gt_scene_graphs.x)
        # [ num_nodes, MAX_OBJ_TOKEN_LEN, sg_emb_dim] -> [ num_nodes, sg_emb_dim]
        x_embed_sum = torch.sum(input=x_embed, dim=-2, keepdim=False)
        # [ num_edges, MAX_EDGE_TOKEN_LEN] -> [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim]
        edge_attr_embed = self.sg_vocab_embedding(gt_scene_graphs.edge_attr)

        # yanhao: for the manually added symmetric edges, reverse the sign of emb to denote reverse relationship:
        edge_attr_embed[gt_scene_graphs.added_sym_edge, :, :] *= -1


        # [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim] -> [ num_edges, sg_emb_dim]
        edge_attr_embed_sum   = torch.sum(input=edge_attr_embed, dim=-2, keepdim=False)
        del x_embed, edge_attr_embed

        ##################################
        # Call scene graph encoding layer
        ##################################
        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoding_layer(
            x=x_embed_sum,
            edge_index=gt_scene_graphs.edge_index,
            edge_attr=edge_attr_embed_sum,
            u=None,
            batch=gt_scene_graphs.batch
            )

        x_encoded = self.graph_layer_norm(x_encoded, gt_scene_graphs.batch)

        return x_encoded, edge_attr_encoded, None


"""
The whole Pipeline. put everything here
"""

class GraphVQAEncoder(torch.nn.Module):
    def __init__(self, args):
        super(GraphVQAEncoder, self).__init__()
        
        ##################################
        # Build scene graph encoder
        ##################################
        self.scene_graph_encoder = SGEncoder()

        ##################################
        # Build text embedding
        ##################################
        text_emb_dim = graphVQAConfig.text_emb_dim # 300d glove
        text_pad_idx = graphVQAConfig.pad_token
        text_vocab_size = graphVQAConfig.vocab_size
        self.text_vocab_embedding = torch.nn.Embedding(text_vocab_size, text_emb_dim, padding_idx=text_pad_idx)
        self.text_vocab_embedding.weight.data.copy_(QAVocab.qa_encoding_text.vocab.vectors)
        self.question_hidden_dim = graphVQAConfig.question_hidden_dim

        ##################################
        # Build Question Encoder
        ##################################
        
        self.question_encoder = TransformerQuestionEncoder(
            text_vocab_embedding=self.text_vocab_embedding,
            text_emb_dim=text_emb_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=graphVQAConfig.no_encoder_attn_heads, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=graphVQAConfig.no_encoder_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=graphVQAConfig.dropout, # the dropout value
            )
        
        '''
        
        self.instr_encoder = TransformerInstrEncoder(
            text_vocab_embedding=self.text_vocab_embedding,
            text_emb_dim=text_emb_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=graphVQAConfig.no_encoder_attn_heads, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=graphVQAConfig.no_encoder_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=graphVQAConfig.dropout, # the dropout value
            )
        '''
        
        ##################################
        # Build Program Decoder
        ##################################
        self.program_decoder = TransformerProgramDecoder(
            text_vocab_embedding=self.text_vocab_embedding,
            vocab_size=text_vocab_size,
            text_emb_dim=text_emb_dim, # embedding dimension
            ninp=self.question_hidden_dim, # transformer encoder layer input dim
            nhead=graphVQAConfig.no_program_decoder_attn_heads, # the number of heads in the multiheadattention models
            nhid=4*self.question_hidden_dim, # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers=graphVQAConfig.no_program_decoder_layers, # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            dropout=graphVQAConfig.dropout, # the dropout value
            )

        ##################################
        # Build Neural Execution Module Pooling Layer
        ##################################


        # input to the gat_seq would be: 
        # 1. concat(h_prev, x_orig), where h_prev is the previous GAT layer's output and x_orig is the original encoded node features
        # 2. concat(edge_attr, ins_i), concat of edge_attr and i_th step instruction vector

        self.gat_seq = gat_seq(in_channels=self.scene_graph_encoder.sg_emb_dim,
                 out_channels=self.scene_graph_encoder.sg_emb_dim, 
                 edge_attr_dim=self.scene_graph_encoder.sg_emb_dim, 
                 ins_dim=self.question_hidden_dim, num_ins=graphVQAConfig.max_execution_step,
                 dropout=graphVQAConfig.dropout, gat_heads=graphVQAConfig.no_gat_attn_heads, 
                 gat_bias=True) # the drop-out is for both dropout in between GATs and dropout inside the GATs



        ##################################
        # Build Neural Execution Module Pooling Layer
        ##################################
        self.graph_global_attention_pooling = ConditionalGlobalAttention(
            num_node_features=self.scene_graph_encoder.sg_emb_dim,
            num_out_features=self.question_hidden_dim)


        return
    
    def forward(self,
                questions,
                gt_scene_graphs,
                programs_input,
                val=False,
                ):

        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoder(gt_scene_graphs)

        ##################################
        # Encode questions
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        questions_encoded = self.question_encoder(questions)

        ##################################
        # Decode programs
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        #instr_vectors = self.instr_encoder(programs_input)
        
        if not val:
            programs_output, instr_vectors = self.program_decoder(memory=questions_encoded, tgt=programs_input)
        else:
            programs_output, instr_vectors = self.program_decoder.sample(memory=questions_encoded, tgt=programs_input)

        ##################################
        # Call Recurrent Neural Execution Module
        ##################################

        x_executed = self.gat_seq(x=x_encoded, edge_index=gt_scene_graphs.edge_index, edge_attr=edge_attr_encoded, instr_vectors=instr_vectors, batch=gt_scene_graphs.batch)

        ##################################
        # Final Layer of the Neural Execution Module, global pooling
        # (batch_size, channels)
        ##################################
        global_language_feature = questions_encoded[0] # should be changed when completing NEM
        graph_final_feature = self.graph_global_attention_pooling(
            x = x_executed, # x=x_encoded,
            u = global_language_feature,
            batch = gt_scene_graphs.batch,
            # no need for edge features since it is global node pooling
            size = None)

        ##################################
        # Call Short Answer Classification Module Only for Debug
        ##################################
        questions_feature = questions_encoded[0]
        short_answer_feature = torch.cat((graph_final_feature, questions_feature, graph_final_feature * questions_feature), dim=-1)
        if not val:
            max_step = graphVQAConfig.max_execution_step
            bs = programs_output.size(1) // max_step
            n_tokens = programs_output.size(0)
            # max_step*batch_size, no_tokens
            programs_output = programs_output.permute(1, 0, 2)
            programs_output = programs_output.reshape((bs, max_step, n_tokens, programs_output.size(2)))
            return short_answer_feature, programs_output, graph_final_feature, questions_feature
        return short_answer_feature, graph_final_feature, questions_feature

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(GraphVQAEncoder, self).load_state_dict(model_dict)
