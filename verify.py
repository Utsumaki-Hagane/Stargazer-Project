"""Independent dataset labels, split audit, and every saved training checkpoint."""
import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from logic_v1.tasks import signatures


def reference(row):
    if row[62]==9: return dict(zip(row[:16:2],row[1:16:2]))[row[63]]
    facts={(row[i],row[i+1]):row[i+2] for i in range(0,48,3)}
    node=row[63]
    for relation in row[57:63]:
        if relation: node=facts[relation,node]
    return node


def main():
    torch.set_num_threads(4)
    report=json.loads(Path('LOGIC_V1_RESULTS.json').read_text()); assert report['complete']
    root=Path(report['run']); config=report['config']; fact_set=set(); training_count=0
    for seed in range(3):
        file=root/f'data/train-{seed}.pt'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==config['data']['dataset_hashes'][file.name]
        d=torch.load(file,map_location='cpu',weights_only=True)
        x=d['inputs'].reshape(-1,64); y=d['targets'].reshape(-1); task=d['tasks'].reshape(-1)
        assert x.shape==(160000,64)
        expected=torch.tensor([reference(row) for row in x.tolist()])
        assert torch.equal(expected,y.long()),seed
        assert torch.equal(torch.bincount(task.long(),minlength=5),torch.full((5,),32000))
        fact_set.update(signatures(x.long(),task)); training_count+=len(y)
        print('verified training labels',seed,len(y),flush=True)
    saved=set(map(tuple,torch.load(root/'data/training_fact_signatures.pt',weights_only=True)))
    assert saved==fact_set
    validations=torch.load(root/'data/validation.pt',weights_only=True)
    heldout_count=0
    for name,d in validations.items():
        expected=torch.tensor([reference(row) for row in d['inputs'].tolist()])
        assert torch.equal(expected,d['targets'].long()),name
        assert not(set(signatures(d['inputs'].long(),torch.full((len(expected),),d['task'])))&fact_set)
        heldout_count+=len(expected)
    expected_ids={(m,s,t) for m in config['models'] for s in range(3) for t in range(1000,5001,100)}
    assert {(r['model'],r['seed'],r['step']) for r in report['rows']}==expected_ids
    for index,row in enumerate(report['rows']):
        file=Path(row['checkpoint'])
        assert file.exists()
        checkpoint=torch.load(file,map_location='cpu',weights_only=True)
        assert checkpoint['global_step']==row['step']
        assert checkpoint['dataset_hash']==config['data']['dataset_hashes'][f"train-{row['seed']}.pt"]
        assert max(v['step'].item() for v in checkpoint['optimizer']['state'].values())==row['step']
        assert all(torch.isfinite(p).all() for p in checkpoint['model'].values())
        if index%100==0: print('verified checkpoints',index,flush=True)
    final=json.loads(Path('LOGIC_V1_FINAL_TEST.json').read_text())
    questions=torch.load(root/'data/final_test.pt',weights_only=True)
    dev_facts=set()
    for d in validations.values(): dev_facts.update(signatures(d['inputs'].long(),torch.full((len(d['targets']),),d['task'])))
    for name,d in questions.items():
        assert torch.equal(torch.tensor([reference(row) for row in d['inputs'].tolist()]),d['targets'].long())
        assert not(set(signatures(d['inputs'].long(),torch.full((len(d['targets']),),d['task'])))&(fact_set|dev_facts))
    assert len(final['rows'])==9
    for row in final['rows']:
        for name,ev in row['evaluation'].items():
            pred=torch.tensor(ev['predictions']); actual=questions[name]['targets'].long()
            assert int((pred==actual).sum())==ev['correct']
    audit=dict(verified=True,training_examples=training_count,validation_examples=heldout_count,
        exact_fact_mappings=len(fact_set),checkpoints=369,final_models=9,
        final_questions_per_model=final['questions_per_model'],independent_labels_verified=True,
        train_validation_final_fact_separation_verified=True)
    (root/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
