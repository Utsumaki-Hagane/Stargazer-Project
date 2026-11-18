"""v0.2: Z_t = GRU(single-token QK pooling conditioned on Z_(t-1), Z_(t-1))."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from stargazer import Model,Stargazer


class RecursiveStargazer(Stargazer):
    def __init__(self,d=64,micro=16,stat=4,zdim=32,heads=4):
        super().__init__(d,micro,stat,zdim,heads)
        self.compress=nn.GRUCell(d,zdim)
        self.previous_query=nn.Linear(zdim,d,bias=False)
        self.zdim=zdim

    def update(self,keys,values,previous):
        q=self.query+self.previous_query(previous)
        scores=(keys*q.unsqueeze(-2)).sum(-1)/math.sqrt(keys.shape[-1])
        pooled=(scores.softmax(-1).unsqueeze(-1)*values).sum(-2)
        return self.compress(pooled,previous)

    def forward(self,x,zero_z=False):
        b,t,d=x.shape
        # Current-token statistics only; historical information lives in previous Z.
        u=self.qv(x).view(b,t,self.micro,self.stat)
        keys,values=self.key(u),self.value(u)
        previous=x.new_zeros(b,self.zdim)
        history=[]
        # Split once so backward combines gradients along time in one operation.
        for key_t, value_t in zip(keys.unbind(dim=1), values.unbind(dim=1), strict=True):
            previous=self.update(key_t,value_t,previous)
            if zero_z: previous=previous*0
            history.append(previous)
        z=torch.stack(history,1)
        q=self.read_query(z).view(b,t,self.h,d//self.h).transpose(1,2)
        k,v=self.kv(x).view(b,t,2,self.h,d//self.h).permute(2,0,3,1,4)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(y.transpose(1,2).reshape(b,t,d))

    def step(self,x,state,zero_z=False):
        b,_,d=x.shape
        u=self.qv(x).view(b,self.micro,self.stat)
        previous=state.get('z',x.new_zeros(b,self.zdim))
        z=self.update(self.key(u),self.value(u),previous)
        state['z']=z*0 if zero_z else z
        q=self.read_query(state['z']).view(b,1,self.h,d//self.h).transpose(1,2)
        k,v=self.kv(x).view(b,1,2,self.h,d//self.h).permute(2,0,3,1,4)
        for name,value in [('k',k),('v',v)]:
            state[name]=torch.cat([state[name],value],dim=2) if name in state else value
        y=F.scaled_dot_product_attention(q,state['k'],state['v'],is_causal=False)
        return self.out(y.transpose(1,2).reshape(b,1,d))


def recursive_model():
    m=Model()
    old=m.blocks[0].attn
    new=RecursiveStargazer()
    shared={k:v for k,v in old.state_dict().items() if not k.startswith('compress.')}
    missing,unexpected=new.load_state_dict(shared,strict=False)
    assert not unexpected
    assert all(k.startswith(('compress.','previous_query.')) for k in missing)
    m.blocks[0].attn=new
    return m
