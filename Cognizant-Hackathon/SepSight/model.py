import torch
import torch.nn as nn
import torch.nn.functional as F
import os
try:
  import einops
  from einops import rearrange,reduce,repeat
except ImportError:
  os.system('pip install einops')
  from einops import rearrange,reduce,repeat
import math
from torch.utils.data import DataLoader,Dataset,Sampler,WeightedRandomSampler
from torch.nn.utils.rnn import pad_packed_sequence,pack_sequence
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


class MHA(nn.Module):
  def __init__(self,dim,attention_dropout,num_heads):
    super().__init__()
    self.dim=dim
    self.attention_dropout=attention_dropout
    self.num_heads=num_heads

    self.q=nn.Linear(dim,dim)
    self.k=nn.Linear(dim,dim)
    self.v=nn.Linear(dim,dim)
    self.out=nn.Linear(dim,dim)
  def forward(self,x,position_emb,padding_mask,is_casual=False):
    H=self.num_heads
    #B T D
    assert position_emb is not None
    assert padding_mask is not None

    q=rearrange(self.q(x + position_emb),pattern="B T (D H) -> B H T D",H=H)
    k=rearrange(self.k(x + position_emb),pattern="B T (D H) -> B H T D",H=H)
    v=rearrange(self.v(x),pattern="B T (D H) -> B H T D",H=H)

    attn=F.scaled_dot_product_attention(q,k,v,attn_mask=padding_mask,is_causal=is_casual)
    attn=rearrange(tensor=attn,pattern="B H T D -> B T (H D)")
    attn=self.out(attn)

    return attn
  
class Cross_Attn(nn.Module):
  def __init__(self,dim,attention_dropout,num_heads):
    super().__init__()
    self.dim=dim
    self.attention_dropout=attention_dropout
    self.num_heads=num_heads

    self.k=nn.Linear(dim,dim)
    self.v=nn.Linear(dim,dim)
    self.q=nn.Linear(dim,dim)
    self.out=nn.Linear(dim,dim)

    self.norm=nn.LayerNorm(dim)

  def forward(self,kv,q,q_embedding,k_embedding):
    H=self.num_heads

    k=rearrange(tensor=self.k(kv + k_embedding),pattern="B T (D H) -> B H T D",H=H)

    v=rearrange(tensor=self.v(kv),pattern="B T (D H) -> B H T D",H=H)

    q=rearrange(tensor=self.q(q + q_embedding),pattern="B T (D H) -> B H T D",H=H)
    attn=F.scaled_dot_product_attention(q,k,v,is_causal=False)
    attn=rearrange(tensor=attn,pattern="B H T D -> B T (H D)")
    attn=self.out(attn)

    return rearrange(tensor=q,pattern="B H T D-> B T (H D)")+self.norm(attn)

class MLP(nn.Module):
  def __init__(self,dim):
    super().__init__()
    self.dim=dim
    self.net=nn.Sequential(
        nn.Linear(dim,dim*2),
        nn.GELU(),
        nn.Linear(dim*2,dim)
    )
  def forward(self,x):
    return self.net(x)

class Add_Norm(nn.Module):
  def __init__(self,module,dim):
    super().__init__()
    self.module=module
    self.dim=dim
    self.ln=nn.LayerNorm(dim)

  def forward(self,x,*args,**kwargs):
    return x + self.ln(self.module(x,*args,**kwargs))
  
class Encoder_layer(nn.Module):
  def __init__(self,dim,n_heads,attn_drop=0.):
    super().__init__()
    self.MHA=Add_Norm(MHA(dim,attn_drop,num_heads=n_heads),dim)
    self.ffn=Add_Norm(MLP(dim),dim)

  def forward(self,x,position_emb,padding_mask):
    x=self.MHA(x,position_emb,padding_mask)
    x=self.ffn(x)
    return x
  
class Decoder_layer(nn.Module):
  def __init__(self,dim,n_heads,attn_drop=0.,first=False):
    super().__init__()
    self.MMHA= nn.Identity() if first else Add_Norm(MHA(dim,attn_drop,n_heads),dim)
    self.cross_attn = Cross_Attn(dim,attn_drop,n_heads)
    self.ffn=Add_Norm(MLP(dim),dim)

  def forward(self,dec_input,enc_input,
              q_embedding,k_embedding,k_d_embedding):

    dec_out=self.MMHA(dec_input,q_embedding,True)
    mlp_out=self.cross_attn(enc_input,q=dec_out,q_embedding=q_embedding,k_embedding=k_embedding)
    out=self.ffn(mlp_out)

    return out

class MMHA_normal(nn.Module):
  def __init__(self,dim,n_heads,attn_drop=0):
    super().__init__()
    self.dim=dim
    self.attention_dropout=attn_drop
    self.num_heads=n_heads

    self.q=nn.Linear(dim,dim)
    self.k=nn.Linear(dim,dim)
    self.v=nn.Linear(dim,dim)
    self.out=nn.Linear(dim,dim)

  def forward(self,x):
    print('Masked MultiHead Attention.........')
    print('Input feature Shape')
    print(x.shape)
    H=self.num_heads

    k=rearrange(tensor=self.k(x),pattern="B T (D H) -> B H T D",H=H)
    print('KEYSS')
    print(k.shape)

    v=rearrange(tensor=self.v(x),pattern="B T (D H) -> B H T D",H=H)
    print('VALUES')
    print(v.shape)

    q=rearrange(tensor=self.q(x),pattern="B T (D H) -> B H T D",H=H)
    print('Queries')
    print(q.shape)
    attn=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    print('After scaled Dot')
    print(attn.shape)
    attn=rearrange(tensor=attn,pattern="B H T D -> B T (H D)")
    print('Rearrange in the MMHA')
    print(attn.shape)
    attn=self.out(attn)
    print('Final output')
    print(attn.shape)

    return attn
class Decoder_block_normal(nn.Module):
  def __init__(self,dim,n_heads,attn_drop=0.,first=False):
    super().__init__()
    self.MMHA= nn.Identity() if first else Add_Norm(MMHA_normal(dim,n_heads,attn_drop),dim)
    self.ffn=Add_Norm(MLP(dim),dim)

  def forward(self,x):
    x=self.MMHA(x)
    x=self.ffn(x)
    return x
  
class Embedding_layer(nn.Module):

  def __init__(self,
               num_features,
               tokens_count,
               dim,
               padding_index=0,
               learnable_timestep=False,time_step_count=335,device="cpu"):
    super().__init__()
    self.device=device
    self.embedding_layer=nn.Embedding(tokens_count+1,embedding_dim=dim,padding_idx=padding_index)

    self.learnable_timestep=learnable_timestep
    if learnable_timestep:
      self.time_step_embedding=nn.Embedding(time_step_count,embedding_dim=dim,padding_idx=padding_index)

    self.conv1=nn.Conv1d(num_features,1,kernel_size=1)

    self.apply(self._init_weight)

  def forward(self,x):
    features=self.embedding_layer(x)
    print('STARTERRRRR')
    print(features.shape)
    if len(features.shape)==3:
      features = features.unsqueeze(0)
    print(features.shape)
    print(features)
    B,T,F,D=features.shape
    features=rearrange(tensor=features,pattern="B T F D-> (B T) F D")

    features=self.conv1(features)
    features= rearrange(tensor=features,pattern="(B T) F D-> B (T F) D",B=B,T=T,F=1)

    if self.learnable_timestep:
      time_step=self.time_step_embedding(x) #TODO implement
    else:
      time_step=self.Sinusoidal_position_Emdedding(length=T,d_model=D).to(self.device)

    return features + time_step,time_step

  def _init_weight(self,module):

    if isinstance(module,nn.Embedding):
      std=0.02
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    if isinstance(module,nn.Conv1d):
      module.weight= nn.init.normal_(module.weight,mean=0.0,std=0.02)
      if module.bias is not None:
        module.bias=nn.init.zeros_(module.bias)

  @staticmethod
  def Sinusoidal_position_Emdedding(length=1024,d_model=768,learnable=False):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return nn.Parameter(pe) if learnable else pe


class Sepsis_Transformer_decoderonly(nn.Module):
  def __init__(self, num_features,
               tokens_count,
               padding_index=0,
               n_heads=4,
               n_layers=2,
               attn_drop=0.,
               dim=128,
               num_classes=2,device="cpu"):
    super().__init__()
    self.device=device
    self.n_heads=n_heads
    self.n_layers=n_layers
    self.attn_drop=attn_drop

    self.embedding_layer=Embedding_layer(
               num_features,
               tokens_count,
               dim,
               padding_index,device=self.device).to(self.device)

    self.decoder_network=nn.ModuleList([
          Decoder_block_normal(dim,n_heads) for _ in range(n_layers)
      ]).to(self.device)


    self.classification= nn.Sequential(
          nn.Linear(dim,num_classes)
      ).to(self.device)
    self.apply(self._init_layers)

  def forward(self,x):

    x,_=self.embedding_layer(x)
    print('Tensor Shape after embedding')
    print(x.shape)

    B,T,D=x.shape

    for i in range(self.n_layers):
      x = self.decoder_network[i](x)

    x=x.view(-1,D)

    x=self.classification(x)

    return x

  def _init_layers(self,module):
    if isinstance(module, nn.Linear):
            std = 0.02
            std*=(2*self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class Sepsis_Transformer(nn.Module):

  def __init__(self,
               num_features,
               tokens_count,
               padding_index=0,
               n_heads=4,
               n_encoder_layers=2,
               n_decoder_layers=2,
               attn_drop=0.,
               dim=128,
               mean=False,
               num_classes=2):

    super().__init__()

    self.n_heads=n_heads
    self.n_encoder_layers=n_encoder_layers
    self.n_decoder_layers=n_decoder_layers
    self.attn_drop=attn_drop
    self.mean=mean
    self.embedding_layer=Embedding_layer(
               num_features,
               tokens_count,
               dim,
               padding_index)
    self.learnable_query=nn.Parameter(torch.zeros(20,dim),requires_grad=True)

    self.encoder_network=nn.ModuleList([
        Encoder_layer(dim,n_heads) for _ in range(n_encoder_layers)
    ])

    self.enc_layers=n_encoder_layers

    self.decoder_network=nn.ModuleList([
        Decoder_layer(dim,n_heads) for _ in range(n_decoder_layers)
    ])

    self.dec_layers=n_decoder_layers

    self.classification= nn.Sequential(
        nn.Linear(dim,num_classes)
    )

    self.encoder_network.apply(self._init_enc_layers)
    self.decoder_network.apply(self._init_dec_layers)


  def forward(self,x, padding_mask , final_tokens=None):
    x,pos_emb=self.embedding_layer(x)

    B,T,D=x.shape

    for i in range(self.n_encoder_layers):

      x=self.encoder_network[i](x,position_emb=pos_emb,padding_mask=padding_mask)

    encoder_out=x
    dec_in=repeat(self.learnable_query,pattern="T D -> B T D",B=B)
    for j in range(self.n_decoder_layers):
      """self,dec_input,enc_input,
              q_embedding,k_embedding,k_d_embedding"""
      dec_in=self.decoder_network[i](dec_input=dec_in,enc_input=encoder_out,
                                     q_embedding=dec_in,k_d_embedding=dec_in,k_embedding=pos_emb)

    if not self.mean:
      final=final(-1,D)
    else:
      final=dec_in.mean(dim=1)

    classes = self.classification(final)

    return classes

  def _init_enc_layers(self,module):
    if isinstance(module, nn.Linear):
            std = 0.02
            std*=(2*self.enc_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

  def _init_dec_layers(self,module):
    if isinstance(module,nn.Linear):
            std = 0.02
            std*=(2*self.dec_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

def create_model():
  model = Sepsis_Transformer_decoderonly(15,123,n_heads=4,attn_drop=0.1)
  return model 