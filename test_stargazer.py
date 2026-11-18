import unittest
import torch
from stargazer import Model, batch, loss_for


class Tests(unittest.TestCase):
    def test_incremental_equivalence_and_single_z(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for kind in ['transformer','stargazer']:
                for zero_z in [False,True]:
                    torch.manual_seed(9)
                    m=Model(kind).to(device).eval()
                    x=torch.randint(0,32,(2,24),device=device)
                    state=None; outputs=[]
                    with torch.no_grad():
                        expected=m(x,zero_z)
                    for t in range(x.shape[1]):
                        logits,state=m.step(x[:,t:t+1],state,zero_z)
                        outputs.append(logits)
                        self.assertEqual(state['position'],t+1)
                        for layer in state['layers']:
                            self.assertEqual(layer['k'].shape[2],t+1)
                        if kind=='stargazer':
                            sg=state['layers'][0]
                            self.assertEqual(sg['count'],t+1)
                            self.assertEqual(tuple(sg['z'].shape),(2,1,32))
                            self.assertEqual(set(sg),{'sum','count','z','k','v'})
                            self.assertFalse(sg['z'].requires_grad)
                    torch.testing.assert_close(torch.cat(outputs,1),expected,atol=2e-5,rtol=2e-5)
                    first,new_state=m.step(x[:,:1])
                    self.assertEqual(new_state['position'],1)
                    torch.testing.assert_close(first,expected[:,:1],atol=2e-5,rtol=2e-5)

    def test_incremental_guards(self):
        m=Model(length=2)
        x=torch.ones(1,1,dtype=torch.long)
        with self.assertRaises(RuntimeError): m.step(x)
        m.eval()
        with self.assertRaises(ValueError): m.step(x.repeat(1,2))
        _,s=m.step(x)
        _,s=m.step(x,s)
        with self.assertRaises(ValueError): m.step(x,s)

    def test_causality_and_prefix(self):
        torch.set_num_threads(2)
        for kind in ['transformer','stargazer']:
            torch.manual_seed(7)
            m=Model(kind).eval()
            x=torch.randint(0,32,(2,32))
            changed=x.clone(); changed[:,17:]=torch.randint(0,32,(2,15))
            with torch.no_grad():
                a,b=m(x),m(changed)
                torch.testing.assert_close(a[:,:17],b[:,:17],atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(a[:,:17],m(x[:,:17]),atol=1e-5,rtol=1e-5)

    def test_no_future_gradient(self):
        m=Model().eval()
        captured=[]
        h=m.emb.register_forward_hook(lambda mod,inp,out:(out.retain_grad(),captured.append(out)) and None)
        m(torch.randint(0,32,(1,24)))[:,10].square().sum().backward()
        self.assertEqual(captured[0].grad[:,11:].abs().sum().item(),0)
        h.remove()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_causality_and_backward(self):
        m=Model().cuda().eval()
        x=torch.randint(0,32,(2,32),device='cuda')
        altered=x.clone(); altered[:,16:]=(altered[:,16:]+1)%32
        torch.testing.assert_close(m(x)[:,:16],m(altered)[:,:16],atol=1e-5,rtol=1e-5)
        m(x).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_learning_and_all_stage_gradients(self):
        torch.manual_seed(0)
        m=Model(); opt=torch.optim.AdamW(m.parameters(),lr=0.003)
        x,y=batch(11,16,32)
        before=loss_for(m,x,y)[0].item()
        for _ in range(25):
            opt.zero_grad(); loss=loss_for(m,x,y)[0]; loss.backward()
            for name in ['qv','key','value','compress','read_query','kv']:
                grads=[p.grad for n,p in m.blocks[0].attn.named_parameters() if n.startswith(name)]
                self.assertTrue(any(g is not None and g.abs().sum()>0 for g in grads),name)
            nnorm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        self.assertLess(loss_for(m,x,y)[0].item(),before*0.8)

    def test_data_labels(self):
        x,y=batch(1,10,32)
        for row,label in zip(x,y):
            i=(row[:16:2]==row[-1]).nonzero().item()
            self.assertEqual(row[2*i+1].item(),label.item())
        x,y=batch(1,3,32,'repeat')
        torch.testing.assert_close(x[:,1:],y[:,:-1])


if __name__=='__main__': unittest.main()
