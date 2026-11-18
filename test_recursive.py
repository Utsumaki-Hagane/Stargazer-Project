import unittest
import torch
from recursive_stargazer import recursive_model


class Tests(unittest.TestCase):
    def test_causality_state_and_gradient(self):
        torch.set_num_threads(2)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(0)
            m=recursive_model().to(device).eval()
            x=torch.randint(0,32,(2,24),device=device)
            other=x.clone(); other[:,12:]=(other[:,12:]+1)%32
            with torch.no_grad():
                expected=m(x)
                torch.testing.assert_close(expected[:,:12],m(other)[:,:12],atol=2e-5,rtol=2e-5)
            state=None; out=[]
            for i in range(24):
                logits,state=m.step(x[:,i:i+1],state)
                out.append(logits)
                self.assertEqual(tuple(state['layers'][0]['z'].shape),(2,32))
                self.assertEqual(set(state['layers'][0]),{'z','k','v'})
            torch.testing.assert_close(torch.cat(out,1),expected,atol=2e-5,rtol=2e-5)
            # Final state must depend on early inputs; full BPTT, no detach.
            a=m.blocks[0].attn
            hidden=torch.randn(2,12,64,device=device,requires_grad=True)
            a(hidden)[:,-1].square().sum().backward()
            self.assertGreater(hidden.grad[:,0].abs().sum().item(),0)
            for part in [a.compress.weight_hh,a.previous_query.weight,a.qv.weight]:
                self.assertTrue(torch.isfinite(part.grad).all())
                self.assertGreater(part.grad.abs().sum().item(),0)
            keys=torch.randn(2,16,64,device=device); values=torch.randn_like(keys)
            self.assertGreater((a.update(keys,values,torch.zeros(2,32,device=device))-a.update(keys,values,torch.ones(2,32,device=device))).abs().sum().item(),0)


if __name__=='__main__': unittest.main()
