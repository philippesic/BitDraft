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
    c['overrides']['blk.0.ffn_gate']['bits']=1
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
