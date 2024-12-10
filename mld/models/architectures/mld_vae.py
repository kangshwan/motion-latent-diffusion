from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""


class MldVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,                           # 263
                 latent_dim: list = [1, 256],           # [1, 256]
                 ff_size: int = 1024,                   # 1024
                 num_layers: int = 9,                   # 9
                 num_heads: int = 4,                    # 4
                 dropout: float = 0.1,                  # 0.1
                 arch: str = "all_encoder",             # encoder_decoder
                 normalize_before: bool = False,        # False
                 activation: str = "gelu",              # gelu
                 position_embedding: str = "learned",   # learned
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]    # 1
        self.latent_dim = latent_dim[-1]    # 256
        input_feats = nfeats                # 263
        output_feats = nfeats               # 263
        self.arch = arch                    # encoder_decoder   
        self.mlp_dist = ablation.MLP_DIST   # False
        self.pe_type = ablation.PE_TYPE     # mld

        if self.pe_type == "actor":
            self.query_pos_encoder = PositionalEncoding(
                self.latent_dim, dropout)
            self.query_pos_decoder = PositionalEncoding(
                self.latent_dim, dropout)
        elif self.pe_type == "mld":
            # 1D position embedding.
            # PositionEmbeddingLearned1D(256)
            self.query_pos_encoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            # PositionEmbeddingLearned1D(256)
            self.query_pos_decoder = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding) 
        else:
            raise ValueError("Not Support PE type")
        
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,    # 256
            num_heads,          # 4
            ff_size,            # 1024
            dropout,            # 0.1
            activation,         # gelu
            normalize_before,   # False
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
                                                            # 9
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,    # 256
                num_heads,          # 4
                ff_size,            # 1024
                dropout,            # 0.1
                activation,         # gelu
                normalize_before,   #False
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        if self.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim))
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            # nn.Parameter(torch.randn(1*2, 256))
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None):
        # Temp
        # Todo
        # remove and test this function
        print("Should Not enter here")

        z, dist = self.encode(features, lengths)
        feats_rst = self.decode(z, lengths)
        return feats_rst, z, dist

    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device
        # 이미 collate_fn으로, nframes는 한 배치안에 있는 가장 긴 모션의 길이
        # nframes는 변할 수 있다. (고정 X)

        # features: [batch_size, nframes, nfeats]
        # lenghts: [batch_size]
        bs, nframes, nfeats = features.shape
        # 각 batchsize에 맞게 mask 생성.
        # motion frame에 대한 mask.
        # [batch_size, max_len]
        # ex): max_len이 196인 경우. . .
        # lengths = [196, 194, 195, ..., 159]
        # tensor([[ True,  True,  True,  ...,  True,  True,  True],
        #         [ True,  True,  True,  ...,  True, False, False],
        #         [ True,  True,  True,  ...,  True,  True, False],
        #         ...,
        #         [ True,  True,  True,  ..., False, False, False],
        #         [ True,  True,  True,  ..., False, False, False],
        #         [ True,  True,  True,  ..., False, False, False]])

        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        # 각 배치에 대해 motion token을 복사

        # shape: [self.latent_size*2, batch_size, self.latent_dim]
        # mu, logvar를 알아야 하기 때문에 self.latent_size * 2
        # self.global_motion_token의 shape: [2, 256]
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        # shape: [batch_size, self.latent_size*2] 
        #        [batch_size, 2]
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        # mask: [batch_size, nframes]
        # shape: [batch_size, self.latent_size * 2]
        #        [batch_size, 2]
    
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        # shape: [self.latent_size*2 + nframes, batch_size, self.latent_dim]
        #        [2 + nframes, batch_size, 256]
        xseq = torch.cat((dist, x), 0)

        if self.pe_type == "actor":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        elif self.pe_type == "mld":
            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]
            # 동일한 shape으로 쭉 통과해서 나옴
            # shape: [2+nframes, batch_size, 256]

            # dist는 생성된 motion token...의 분포 (평균과 분산)
            # query_pos = self.query_pos_encoder(xseq)
            # dist = self.encoder(xseq, pos=query_pos, src_key_padding_mask=~aug_mask)[
            #     : dist.shape[0]
            # ]

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        if self.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            # dist의 앞 2개에 대해서만 slicing을 통해 얻어냄

            # shape: [1, batch_size, 256]
            mu = dist[0:self.latent_size, ...]
            # shape: [1 + nframes, batch_size, 256]
            logvar = dist[self.latent_size:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)

        # shape: [2+nframes, batch_size, 256]
        latent = dist.rsample()
        return latent, dist

    def decode(self, z: Tensor, lengths: List[int]):
        # z의 shape: [1, bs, 256](test) or [1+nframe, bs, 256](in train VAE)
        # mask로 VAE 학습에 있어서 서로 다른 길이를 batch로 복원할 수 있도록 함
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape
        
        # batch가 1이라면 mask는 all true.
        # vae 학습중이라면 nframes는 제공된 nframe.
        # test라면 직접 제공한 nframe 혹은 val data의 nframe.
        # 결국 nframes는 어떤 방식으로든지간에 사람이 직접 제공하거나, 입력으로 제공되어야 한다.
        
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # todo
        # investigate the motion middle error!!!

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, queries), axis=0)
            z_mask = torch.ones((bs, self.latent_size),
                                dtype=bool,
                                device=z.device)
            augmask = torch.cat((z_mask, mask), axis=1)

            if self.pe_type == "actor":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
            elif self.pe_type == "mld":
                xseq = self.query_pos_decoder(xseq)
                output = self.decoder(
                    xseq, src_key_padding_mask=~augmask)[z.shape[0]:]
                # query_pos = self.query_pos_decoder(xseq)
                # output = self.decoder(
                #     xseq, pos=query_pos, src_key_padding_mask=~augmask
                # )[z.shape[0] :]

        elif self.arch == "encoder_decoder":
            if self.pe_type == "actor":
                queries = self.query_pos_decoder(queries)
                output = self.decoder(tgt=queries,
                                      memory=z,
                                      tgt_key_padding_mask=~mask).squeeze(0)
            elif self.pe_type == "mld":
                # shape: [nframes, bs, self.latent_dim]
                queries = self.query_pos_decoder(queries)
                # mem_pos = self.mem_pos_decoder(z)
                output = self.decoder(
                    tgt=queries,
                    memory=z,
                    tgt_key_padding_mask=~mask,
                    # query_pos=query_pos,
                    # pos=mem_pos,
                ).squeeze(0)                # 강승환 - squeeze(0)이 필요할까? 애초에 output의 shape는 [nframes, bs, latent_dim]일텐데..
                                            # 학습할 때 nframes는 40 이상이라며...
                                            # 혹시 추론할 때 문제가 생길수도 있기 때문일까?
                                            # length 1짜리를 생성하라고 했을 때 문제가 발생할 까봐!??
                # query_pos = self.query_pos_decoder(queries)
                # # mem_pos = self.mem_pos_decoder(z)
                # output = self.decoder(
                #     tgt=queries,
                #     memory=z,
                #     tgt_key_padding_mask=~mask,
                #     query_pos=query_pos,
                #     # pos=mem_pos,
                # ).squeeze(0)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats 
