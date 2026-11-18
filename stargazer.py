"""Phase 1 research prototype. All architecture choices are provisional."""
import argparse, copy, hashlib, json, math, platform, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3*d)
        self.out = nn.Linear(d, d)

    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(x).view(b,t,3,self.h,d//self.h).permute(2,0,3,1,4)
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d))

    def step(self,x,state):
        b,_,d=x.shape
        q,k,v=self.qkv(x).view(b,1,3,self.h,d//self.h).permute(2,0,3,1,4)
        for name,value in [('k',k),('v',v)]:
            state[name]=torch.cat([state[name],value],dim=2) if name in state else value
        # Cache contains only past and current positions. A 1xT causal mask
        # would incorrectly hide all but the first key with SDPA alignment.
        y=F.scaled_dot_product_attention(q,state['k'],state['v'],is_causal=False)
        return self.out(y.transpose(1,2).reshape(b,1,d))


class Stargazer(nn.Module):
    def __init__(self, d, micro=16, stat=4, zdim=32, heads=4):
        super().__init__()
        self.micro,self.stat,self.h = micro,stat,heads
        self.qv = nn.Linear(d,micro*stat)
        self.key = nn.Linear(stat,d)
        self.value = nn.Linear(stat,d)
        self.query = nn.Parameter(torch.randn(d)/math.sqrt(d))
        self.compress = nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,zdim),nn.LayerNorm(zdim))
        self.read_query = nn.Linear(zdim,d)
        self.kv = nn.Linear(d,2*d)
        self.out = nn.Linear(d,d)

    def forward(self,x,zero_z=False):
        b,t,d = x.shape
        # Independent micro-head statistics; no content-dependent routing in QV.
        u = self.qv(x).view(b,t,self.micro,self.stat)
        counts = torch.arange(1,t+1,device=x.device,dtype=x.dtype)[None,:,None,None]
        s = u.cumsum(1)/counts
        scores = (self.key(s)*self.query).sum(-1)/math.sqrt(d)
        pooled = (scores.softmax(-1).unsqueeze(-1)*self.value(s)).sum(-2)
        z = self.compress(pooled)
        if zero_z:
            z = z*0
        q = self.read_query(z).view(b,t,self.h,d//self.h).transpose(1,2)
        k,v = self.kv(x).view(b,t,2,self.h,d//self.h).permute(2,0,3,1,4)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.out(y.transpose(1,2).reshape(b,t,d))

    def step(self,x,state,zero_z=False):
        b,_,d=x.shape
        u=self.qv(x).view(b,1,self.micro,self.stat)
        state['sum']=state['sum']+u if 'sum' in state else u
        state['count']=state.get('count',0)+1
        s=state['sum']/state['count']
        scores=(self.key(s)*self.query).sum(-1)/math.sqrt(d)
        pooled=(scores.softmax(-1).unsqueeze(-1)*self.value(s)).sum(-2)
        z=self.compress(pooled)
        # Replace the previous conclusion; never append Z history.
        state['z']=z*0 if zero_z else z
        q=self.read_query(state['z']).view(b,1,self.h,d//self.h).transpose(1,2)
        k,v=self.kv(x).view(b,1,2,self.h,d//self.h).permute(2,0,3,1,4)
        for name,value in [('k',k),('v',v)]:
            state[name]=torch.cat([state[name],value],dim=2) if name in state else value
        y=F.scaled_dot_product_attention(q,state['k'],state['v'],is_causal=False)
        return self.out(y.transpose(1,2).reshape(b,1,d))


class Block(nn.Module):
    def __init__(self,d,sg=False,ff=None):
        super().__init__()
        self.n1,self.n2 = nn.LayerNorm(d),nn.LayerNorm(d)
        self.attn = Stargazer(d) if sg else Attention(d)
        self.ff = nn.Sequential(nn.Linear(d,ff or 4*d),nn.GELU(),nn.Linear(ff or 4*d,d))

    def forward(self,x,zero_z=False):
        a = self.attn(self.n1(x),zero_z) if isinstance(self.attn,Stargazer) else self.attn(self.n1(x))
        x = x+a
        return x+self.ff(self.n2(x))

    def step(self,x,state,zero_z=False):
        u=self.n1(x)
        a=self.attn.step(u,state,zero_z) if isinstance(self.attn,Stargazer) else self.attn.step(u,state)
        x=x+a
        return x+self.ff(self.n2(x))


class Model(nn.Module):
    def __init__(self,kind='stargazer',d=64,length=64,ff=None):
        super().__init__()
        self.emb = nn.Embedding(32,d)
        self.pos = nn.Embedding(length,d)
        # One experimental front-end, followed by one common Transformer block.
        self.blocks = nn.ModuleList([Block(d,kind=='stargazer',ff),Block(d)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,32,bias=False)

    def forward(self,x,zero_z=False):
        y = self.emb(x)+self.pos(torch.arange(x.shape[1],device=x.device))
        for block in self.blocks:
            y = block(y,zero_z)
        return self.head(self.norm(y))

    @torch.inference_mode()
    def step(self,x,state=None,zero_z=False):
        """One-token inference. Caller owns state; pass None for a new sequence.

        Only current Z is retained. Raw KV caches grow with prefix length.
        Training remains parallel and differentiable through forward().
        """
        if self.training:
            raise RuntimeError('Call model.eval() before incremental inference')
        if x.ndim!=2 or x.shape[1]!=1:
            raise ValueError('step expects [batch, 1] token IDs')
        if state is None:
            state=dict(position=0,layers=[{} for _ in self.blocks],batch=x.shape[0],zero_z=zero_z)
        if state['batch']!=x.shape[0] or state['zero_z']!=zero_z:
            raise ValueError('Start a new state when batch or ablation changes')
        pos=state['position']
        if pos>=self.pos.num_embeddings:
            raise ValueError('Prefix exceeds configured positional context')
        y=self.emb(x)+self.pos(torch.tensor([pos],device=x.device))
        for block,layer in zip(self.blocks,state['layers']):
            y=block.step(y,layer,zero_z)
        state['position']=pos+1
        return self.head(self.norm(y)),state


def batch(seed,batch_size,length,task='recall'):
    g = torch.Generator().manual_seed(seed)
    if task=='recall':
        # key IDs 1..8, value IDs 16..31, delimiter 9; no answer in input.
        x = torch.zeros(batch_size,length,dtype=torch.long)
        keys = torch.stack([torch.randperm(8,generator=g)+1 for _ in range(batch_size)])
        vals = torch.randint(16,32,(batch_size,8),generator=g)
        x[:,:16:2],x[:,1:16:2] = keys,vals
        x[:,16:-2] = torch.randint(10,16,(batch_size,length-18),generator=g)
        idx = torch.randint(0,8,(batch_size,),generator=g)
        x[:,-2],x[:,-1] = 9,keys[torch.arange(batch_size),idx]
        return x,vals[torch.arange(batch_size),idx]
    # Next-token task: repeat an eight-token pattern; targets shifted by one.
    p = torch.randint(1,32,(batch_size,8),generator=g)
    seq = p.repeat(1,(length+8)//8)[:,:length+1]
    return seq[:,:-1],seq[:,1:]


def loss_for(model,x,y,zero_z=False):
    logits = model(x,zero_z)
    pred = logits[:,-1] if y.ndim==1 else logits
    return F.cross_entropy(pred.reshape(-1,32),y.reshape(-1)),(pred.argmax(-1)==y).float().mean()


def count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--length',type=int,default=64)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--task',choices=['recall','repeat'],default='recall')
    p.add_argument('--device',default='cuda')
    p.add_argument('--out',default='results')
    a=p.parse_args()
    if a.length<18 or a.steps<1:
        p.error('length >= 18 and steps >= 1 required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(a.out)/time.strftime('%Y%m%d-%H%M%S')
    root.mkdir(parents=True,exist_ok=False)
    env=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
             device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,
             arguments=vars(a),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    target=count(Model())
    # Match parameter count using useful FFN capacity, never dummy parameters.
    ff=min(range(32,513),key=lambda n:abs(count(Model('transformer',ff=n))-target))
    rows=[]
    for seed in a.seeds:
        for kind,width,ablation in [('transformer',256,False),('transformer_matched',ff,False),('stargazer',256,False),('stargazer_zero_z',256,True)]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            m=Model('stargazer' if kind.startswith('stargazer') else 'transformer',length=a.length,ff=width).to(a.device)
            opt=torch.optim.AdamW(m.parameters(),lr=0.001,weight_decay=0.01)
            xv,yv=batch(900000,256,a.length,a.task)
            xv,yv=xv.to(a.device),yv.to(a.device)
            def evaluate():
                m.eval()
                with torch.no_grad():
                    l,acc=loss_for(m,xv,yv,ablation)
                m.train()
                return l.item(),acc.item()
            initial=evaluate()
            # Separate warmup model; warmup does not change experimental state.
            warm=copy.deepcopy(m)
            wx,wy=batch(800000,a.batch,a.length,a.task)
            wx,wy=wx.to(a.device),wy.to(a.device)
            for _ in range(3):
                loss_for(warm,wx,wy,ablation)[0].backward(); warm.zero_grad(set_to_none=True)
            del warm
            if a.device=='cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            history=[]
            for step in range(a.steps):
                x,y=batch(10000+seed*100000+step,a.batch,a.length,a.task)
                x,y=x.to(a.device),y.to(a.device)
                opt.zero_grad(set_to_none=True)
                loss,_=loss_for(m,x,y,ablation)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                norm=nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
                opt.step()
                if step%50==0: history.append(dict(step=step,loss=loss.item(),grad_norm=norm.item()))
            if a.device=='cuda': torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated() if a.device=='cuda' else None
            final=evaluate()
            row=dict(seed=seed,model=kind,parameters=count(m),initial_loss=initial[0],final_loss=final[0],accuracy=final[1],seconds=elapsed,tokens_per_second=a.steps*a.batch*a.length/elapsed,peak_allocated_bytes=peak,history=history)
            rows.append(row)
            torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),args=vars(a),row=row),root/f'{kind}-{seed}.pt')
            (root/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
            del m,opt
    print('Saved:',root)


if __name__=='__main__':
    main()
