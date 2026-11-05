import copy
import unittest
import torch
from scripts.dflash_baseline.prototype import Projection,resolve,digest,native_admit

def config():
    return dict(version=1,preset='essential-a8',source=dict(path='s',sha256='a'*64),parent=dict(path='p',sha256='b'*64),data=dict(path='d',sha256='c'*64),budget_seconds=1)

def test_config():
    c=config();c['overrides']={'blk.0.ffn_gate':dict(weight='W1',bits=4)}
    r=resolve(c);assert r['coverage']['blk.0.ffn_gate']['bits']==4
    assert r['coverage']['blk.1.ffn_gate']['bits']==8
    assert digest(r)==digest(copy.deepcopy(r))
    c['overrides']['blk.0.ffn_gate']['bits']=3
    with unittest.TestCase().assertRaises(ValueError):resolve(c)
    with unittest.TestCase().assertRaises(ValueError):native_admit(r)

def check_integer_and_midpoint(bits):
    m=Projection(torch.tensor([[1.,-1.,1.]]),bits,quantizer={'mode':'fixed','clip':1.})
    with torch.no_grad():m.midpoint.fill_(0.3)
    x=torch.tensor([[0.,-0.5,5.]])
    codes,step=m.stream(x)
    expected=((codes.to(torch.int32)@torch.tensor([[1],[-1],[1]],dtype=torch.int32)).float()+.3*codes.sum(-1,keepdim=True))*step
    torch.testing.assert_close(m(x),expected)

def test_integer_widths():
    for bits in (2,4,8): check_integer_and_midpoint(bits)

def test_gain_and_gradient_state():
    gain=torch.nn.Parameter(torch.zeros(3))
    a=Projection(torch.tensor([[1.,-1.,1.]]),4,gain=gain,quantizer={'mode':'learned','clip':1.})
    b=Projection(torch.ones(1,3),4,gain=gain)
    assert a.log_gain is b.log_gain
    a(torch.tensor([[2.,.23,-.42]])).sum().backward()
    assert torch.isfinite(a.log_clip.grad) and a.log_clip.grad.abs()>0
    c=Projection(torch.ones(1,3),4,quantizer={'mode':'learned','clip':1.})
    c.load_state_dict(a.state_dict());torch.testing.assert_close(c.log_clip,a.log_clip)

if __name__ == "__main__":
    test_config()
    for bits in (2,4,8): check_integer_and_midpoint(bits)
    test_gain_and_gradient_state()
    print("5 CPU cases passed")


def test_independent_rounding():
    for bits in (2,4,8):
        q=(1<<(bits-1))-1
        m=Projection(torch.ones(1,7),bits,quantizer={'mode':'fixed','clip':float(q)})
        codes,_=m.stream(torch.tensor([[-1000.,-1.5,-.5,0.,.5,1.5,1000.]]))
        expected=torch.tensor([[-q,max(-q,-2),0,0,0,min(q,2),q]],dtype=torch.float32)
        torch.testing.assert_close(codes,expected,rtol=0,atol=0)


def test_original_baseline_codes():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from w1a1_eagle.recurrent_qat import fixed_activation_codes
    x=torch.tensor([[0.,0.,0.],[1e-40,-2e-40,0.],[.5,-1.5,127.]])
    m=Projection(torch.tensor([[.2,-.7,.1]]),8)
    codes,step=m.stream(x)
    old,scale,_=fixed_activation_codes(x,8)
    torch.testing.assert_close(codes,old,rtol=0,atol=0)
    torch.testing.assert_close(step,scale,rtol=0,atol=0)
    torch.testing.assert_close(m.latent_sign,torch.tensor([[.2,-.7,.1]]),rtol=0,atol=0)
    expected=(old@torch.tensor([[1.],[-1.],[1.]]))*m.scale.detach()*scale
    torch.testing.assert_close(m(x),expected,rtol=0,atol=0)


def test_model_install_resume():
    import tempfile
    from pathlib import Path
    from torch import nn
    from scripts.dflash_baseline.prototype_model import install,optimizer,save,resume
    class Old(nn.Module):
        def __init__(self):
            super().__init__();self.latent_sign=nn.Parameter(torch.tensor([[.2,-.7,.1]]))
        def effective_scales(self): return self.latent_sign.detach().abs().mean(-1)
    class Fixture(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.ModuleList()
            for _ in range(6):
                l=nn.Module()
                for name in ('ffn_gate','ffn_up','ffn_down'):setattr(l,name,Old())
                for name in ('attn_norm','attn_q_norm','attn_k_norm','ffn_norm'):setattr(l,name,nn.Parameter(torch.ones(1)))
                self.layers.append(l)
            self.output_norm=nn.Parameter(torch.ones(1));self.output_norm_enc=nn.Parameter(torch.ones(1))
        def binary_linears(self):return {f'blk.{i}.{n}':getattr(l,n) for i,l in enumerate(self.layers) for n in ('ffn_gate','ffn_up','ffn_down')}
    c=resolve(config());m=install(Fixture(),c,{})
    for l in m.layers:
        assert l.ffn_gate.log_gain is l.ffn_up.log_gain
        assert l.ffn_gate.log_gain is not l.ffn_down.log_gain
    opt=optimizer(m)
    params=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(params)==len(set(params))
    def step():
        x=torch.randn(2,3)
        sum(p(x).sum() for p in m.binary_linears().values()).backward();opt.step();opt.zero_grad()
    step()
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'state.pt';progress=dict(updates=1,update_seconds=.1,cursor=1,order=[0,1])
        save(p,m,opt,progress);step();expected={k:v.clone() for k,v in m.state_dict().items()}
        assert resume(p,m,opt)==progress
        step()
        for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
        m.prototype_config=copy.deepcopy(c);m.prototype_config['coverage']['blk.0.ffn_gate']['bits']=4
        with unittest.TestCase().assertRaisesRegex(ValueError,'adaptation'):resume(p,m,opt)


def test_baseline_backward_optimizer():
    from w1a1_eagle.block_qat import BlockBinaryLinear
    from w1a1_eagle.recurrent_qat import W1AxContract
    weight=torch.tensor([[2.,-.7,.1],[-2.,.2,-.1]])
    scale=weight.abs().mean(-1)
    old=BlockBinaryLinear(weight,scale,W1AxContract(activation_bits=8))
    new=Projection(weight,8,midpoint=False)
    a=torch.optim.AdamW(old.parameters(),lr=1e-4,weight_decay=0,foreach=False)
    b=torch.optim.AdamW([new.latent_sign,new.scale],lr=1e-4,weight_decay=0,foreach=False)
    for values in ([[0.,0.,0.]],[[1e-40,-2e-40,0.]],[[.5,-1.5,127.]]):
        x=torch.tensor(values,requires_grad=True);y=x.detach().clone().requires_grad_()
        old(x).sum().backward();new(y).sum().backward()
        for actual,expected in ((y.grad,x.grad),(new.latent_sign.grad,old.latent_sign.grad),(new.scale.grad,old.scale_offset.grad)):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a.step();b.step();a.zero_grad();b.zero_grad()
        torch.testing.assert_close(new.latent_sign,old.latent_sign,rtol=0,atol=0)
        torch.testing.assert_close(new.scale,old.effective_scales(),rtol=1e-6,atol=1e-7)
