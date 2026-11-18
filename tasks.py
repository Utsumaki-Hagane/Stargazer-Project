"""Original recall plus F^(h-1) then G relational lookup, original vocabulary."""
import torch
from stargazer import batch as old_batch

TASKS=['recall','chain1','chain2','chain3','chain4','chain5_ood','chain6_ood']


def graph_batch(seed,n,hops,stress=False):
    g=torch.Generator().manual_seed(seed)
    nodes=torch.arange(1,9).expand(n,-1)
    order=torch.rand(n,8,generator=g).argsort(1)
    cycle=torch.empty(n,8,dtype=torch.long)
    cycle.scatter_(1,order,(order.roll(-1,1)+1))
    random_f=torch.rand(n,8,generator=g).argsort(1)+1
    f=torch.where((torch.rand(n,1,generator=g)<.5)|stress,cycle,random_f)
    distinct_g=torch.rand(n,16,generator=g).argsort(1)[:,:8]+16
    random_g=torch.randint(16,32,(n,8),generator=g)
    values=torch.where((torch.rand(n,1,generator=g)<.5)|stress,distinct_g,random_g)
    triples=torch.cat([torch.stack([torch.full_like(nodes,10),nodes,f],-1),
                       torch.stack([torch.full_like(nodes,11),nodes,values],-1)],1)
    shuffle=torch.rand(n,16,generator=g).argsort(1)
    triples=triples.gather(1,shuffle[:,:,None].expand(-1,-1,3))
    x=torch.zeros(n,64,dtype=torch.long); x[:,:48]=triples.reshape(n,48)
    x[:,48:56]=torch.randint(12,16,(n,8),generator=g)
    x[:,56]=9; x[:,62]=11
    if isinstance(hops,int): hops=torch.full((n,),hops,dtype=torch.long)
    assert hops.min()>=1 and hops.max()<=6
    for j in range(5): x[:,57+j]=torch.where(hops>=6-j,10,0)
    start=torch.randint(1,9,(n,),generator=g); x[:,-1]=start
    current=start-1
    for j in range(5): current=torch.where(hops>j+1,f.gather(1,current[:,None]).squeeze(1)-1,current)
    target=values.gather(1,current[:,None]).squeeze(1)
    return x,target


def training_batch(seed,step,n=32):
    task=(torch.arange(n)+step*n)%5
    hops=task.clamp(min=1)
    x,y=graph_batch(80000000+seed*100000+step,n,hops)
    old_x,old_y=old_batch(100000000+seed*100000+step,n,64)
    x[task==0]=old_x[task==0]; y[task==0]=old_y[task==0]
    return x,y,task


def signatures(x,task):
    """Exact canonical fact mapping: kind plus <=56-bit integer."""
    result=torch.zeros(len(x),dtype=torch.long)
    old=task==0
    if old.any():
        rows=x[old]; values=rows[:,1:16:2].gather(1,rows[:,:16:2].argsort(1))-16
        result[old]=(values*(16**torch.arange(8))).sum(1)
    if (~old).any():
        triples=x[~old,:48].reshape(-1,16,3)
        outputs=[]
        for rel in [10,11]:
            selected=triples[triples[:,:,0]==rel].reshape(-1,8,3)
            outputs.append(selected[:,:,2].gather(1,selected[:,:,1].argsort(1)))
        f,values=outputs
        result[~old]=((f-1)*(8**torch.arange(8))).sum(1)+((values-16)*(16**torch.arange(8))).sum(1)*(2**24)
    return list(zip((~old).long().tolist(),result.tolist()))


def oracle(row):
    if int(row[-2])==9:
        facts=dict(zip(row[:16:2].tolist(),row[1:16:2].tolist()))
        return facts[int(row[-1])]
    assert int(row[56])==9
    triples=row[:48].reshape(16,3).tolist()
    facts={(r,a):b for r,a,b in triples}
    current=int(row[-1])
    operations=[int(v) for v in row[57:63] if int(v)!=0]
    assert operations[-1]==11 and all(v==10 for v in operations[:-1])
    for rel in operations: current=facts[rel,current]
    return current


def evaluation_set(seed,n,task,forbidden,stress=False):
    accepted=[]; targets=[]; seen=set(); attempt=0
    while len(accepted)<n:
        if task==0: x,y=old_batch(seed+attempt,128,64)
        else: x,y=graph_batch(seed+attempt,128,task,stress)
        codes=signatures(x,torch.full((128,),task))
        for i,code in enumerate(codes):
            if code in forbidden or code in seen: continue
            seen.add(code); accepted.append(x[i]); targets.append(y[i])
            if len(accepted)==n: break
        attempt+=1
    return dict(inputs=torch.stack(accepted).to(torch.uint8),targets=torch.stack(targets).to(torch.uint8),task=task,stress=stress)
