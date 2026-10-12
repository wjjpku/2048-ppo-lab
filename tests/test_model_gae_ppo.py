from dataclasses import replace
import numpy as np
import pytest
import torch
from game2048.config import Config
from game2048.model import ActorCritic, encode_boards, masked_log_probs, categorical_entropy, exact_kl, sample_actions
from game2048.gae import compute_gae, normalize_advantages
from game2048.ppo import ppo_losses, make_optimizer, ppo_update, rollout_kl
from game2048.runtime import setup, cpu_device_parity
from game2048.rollout import Rollout


def test_model_parameters_encoding_and_init():
    setup()
    model=ActorCritic()
    assert sum(p.numel() for p in model.parameters())==153349
    boards=torch.arange(32).reshape(2,4,4)%21
    onehot=encode_boards(boards)
    assert onehot.shape==(2,336) and onehot.dtype==torch.float32
    assert torch.equal(onehot.reshape(2,16,21).argmax(-1),boards.reshape(2,16))
    logits,v=model(onehot)
    assert logits.shape==(2,4) and v.shape==(2,)
    for module,gain in [(model.trunk[0],2.),(model.trunk[2],2.),(model.policy,.0001),(model.value,1.)]:
        torch.testing.assert_close(module.weight@module.weight.T,torch.eye(len(module.weight))*gain,atol=2e-6,rtol=1e-4)
        assert torch.count_nonzero(module.bias)==0
    with pytest.raises(ValueError): encode_boards(torch.full((1,4,4),21))
    with pytest.raises(TypeError): encode_boards(torch.zeros(1,4,4))


def test_masks_entropy_kl_and_finite_gradients():
    masks=torch.tensor([[bool(i&(1<<j)) for j in range(4)] for i in range(1,16)])
    logits=torch.randn(15,4,requires_grad=True)
    lp=masked_log_probs(logits,masks)
    assert torch.isneginf(lp[~masks]).all()
    torch.testing.assert_close(lp.exp().sum(-1),torch.ones(15))
    assert (lp.exp()[~masks]==0).all()
    entropy=categorical_entropy(lp,masks)
    assert torch.equal(entropy[masks.sum(-1)==1],torch.zeros(4))
    assert torch.equal(exact_kl(lp,lp,masks),torch.zeros(15))
    (entropy.sum()+exact_kl(lp.detach(),lp,masks).sum()).backward()
    assert torch.isfinite(logits.grad).all()
    generator=torch.Generator().manual_seed(13)
    for _ in range(20):
        action=sample_actions(lp.detach(),generator)
        assert masks.gather(1,action[:,None]).all()
    with pytest.raises(ValueError): masked_log_probs(torch.zeros(1,4),torch.zeros(1,4,dtype=torch.bool))
    with pytest.raises(FloatingPointError): masked_log_probs(torch.full((1,4),float('nan')),torch.ones(1,4,dtype=torch.bool))
    with pytest.raises(FloatingPointError): masked_log_probs(torch.tensor([[3e38,-3e38,0.,0.]]),torch.ones(1,4,dtype=torch.bool))


def test_gae_exact_boundary_cases_and_rollout_tail():
    r=torch.ones(1,4);v=torch.full((1,4),2.);nv=torch.full((1,4),7.)
    term=torch.tensor([[True,False,False,True]])
    trunc=torch.tensor([[False,True,False,True]])
    a,ret=compute_gae(r,v,nv,term,trunc)
    torch.testing.assert_close(a,torch.tensor([[-1.,5.993,5.993,-1.]]))
    torch.testing.assert_close(ret,a+v)
    # Ordinary first transition includes next advantage exactly; truncation isolates reset.
    r=torch.tensor([[1.],[7.]])
    v=torch.full((2,1),2.);nv=torch.tensor([[7.],[0.]])
    a,_=compute_gae(r,v,nv,torch.tensor([[False],[True]]),torch.zeros(2,1,dtype=torch.bool))
    assert float(a[0])==pytest.approx(10.73825,abs=1e-6)
    a,_=compute_gae(r,v,nv,torch.tensor([[False],[True]]),torch.tensor([[True],[False]]))
    assert float(a[0])==pytest.approx(5.993,abs=1e-6)


def test_advantage_normalization_targets_unchanged():
    a=torch.tensor([[1.,2.],[3.,4.]])
    old=a.clone()
    norm,stats=normalize_advantages(a)
    assert torch.equal(a,old)
    assert float(norm.mean())==pytest.approx(0.,abs=1e-7)
    assert float(norm.std(unbiased=False))==pytest.approx(1.)
    constant,stats=normalize_advantages(torch.ones(2,2))
    assert stats['advantage_near_zero_std'] and torch.equal(constant,torch.zeros_like(constant))


def test_ppo_signed_clipping_value_coefficient_and_optimizer():
    c=Config()
    ratios=torch.tensor([1.5,.5,1.5,.5])
    a=torch.tensor([1.,1.,-1.,-1.])
    total,m=ppo_losses(ratios.log(),torch.zeros(4),a,torch.full((4,),2.),torch.zeros(4),torch.ones(4),c)
    expected=-torch.tensor([1.2,.5,-1.5,-.8]).mean()
    torch.testing.assert_close(m['policy_loss'],expected)
    assert float(m['value_loss'])==4.
    assert float(total)==pytest.approx(float(expected)+.5*4-.01)
    opt=make_optimizer(ActorCritic(),c)
    assert opt.defaults['eps']==1e-5 and opt.defaults['weight_decay']==0
    assert opt.defaults['lr']==3e-4 and opt.defaults['betas']==(.9,.999)


def test_update_targets_immutable_identity_kl_and_stop():
    setup()
    c=replace(Config(),num_envs=2,rollout_steps=4,minibatch_size=4,epochs=4,target_kl=1e-12)
    model=ActorCritic(); opt=make_optimizer(model,c)
    board=torch.zeros(4,2,4,4,dtype=torch.uint8)
    board[:,:,0,0]=1
    obs=encode_boards(board)
    masks=torch.ones(4,2,4,dtype=torch.bool)
    with torch.no_grad():
        logits,v=model(obs);lp=masked_log_probs(logits,masks)
        action=torch.zeros(4,2,dtype=torch.long)
        d={'obs_onehot':obs,'legal_mask':masks,'old_log_probs':lp,'old_logprob':lp[:,:,0],
           'action':action,'old_value':v,'return':v+2.,'policy_advantage':torch.ones(4,2)}
    before={k:x.clone() for k,x in d.items()}
    rollout=Rollout(d,0,[],{}, {})
    assert rollout_kl(model,rollout.flatten(),4)==0.
    result=ppo_update(model,opt,rollout,c,torch.Generator().manual_seed(5))
    assert result['epochs']==1 and result['optimizer_steps']==2
    assert result['grad_norm_after']<=.50001
    assert result['exact_kl']>c.target_kl
    assert all(torch.equal(before[k],x) for k,x in d.items())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_cpu_parity():
    assert cpu_device_parity('cpu')['passed']
