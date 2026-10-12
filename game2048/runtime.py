import os, time, random
import numpy as np
import torch
from .model import ActorCritic, encode_boards, masked_log_probs, categorical_entropy, exact_kl

def setup(seed=1, threads=1, deterministic=True, device='cpu'):
    if torch.device(device).type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not torch.cuda.is_available(): raise RuntimeError('requested CUDA is unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.device(device).type == 'cuda': torch.cuda.manual_seed_all(seed)
    sample_rng = torch.Generator(device=device).manual_seed(seed+1000003)
    shuffle_rng = torch.Generator(device='cpu').manual_seed(seed+2000003)
    return sample_rng, shuffle_rng

def cpu_device_parity(device='cuda:0', atol=2e-5, rtol=2e-4, hidden_size=256):
    """Fixed small forward/loss parity; no training result is inferred from this."""
    from .config import Config
    from .ppo import ppo_losses
    start = time.monotonic()
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
        torch.manual_seed(123)
        cpu=ActorCritic(hidden_size).eval()
        other=ActorCritic(hidden_size).to(device).eval(); other.load_state_dict(cpu.state_dict())
        b=torch.arange(32*16,dtype=torch.int64).reshape(32,4,4)%16
        masks=torch.tensor([[True,False,True,True],[False,True,False,False]]*16)
        x=encode_boards(b)
        c_logits,c_v=cpu(x); d_logits,d_v=other(x.to(device))
        c_lp=masked_log_probs(c_logits,masks); d_lp=masked_log_probs(d_logits,masks.to(device))
        c_h=categorical_entropy(c_lp,masks); d_h=categorical_entropy(d_lp,masks.to(device))
        a=masks.int().argmax(-1);adv=torch.linspace(-2,2,32);returns=torch.linspace(-1,3,32)
        old=c_lp.gather(1,a[:,None]).squeeze(1).detach()
        cl,_=ppo_losses(old,old,adv,c_v,returns,c_h,Config())
        dl,_=ppo_losses(d_lp.gather(1,a.to(device)[:,None]).squeeze(1),old.to(device),adv.to(device),d_v,returns.to(device),d_h,Config())
        for c,d in [(c_logits,d_logits),(c_v,d_v),(c_h,d_h),(cl,dl)]:
            torch.testing.assert_close(c,d.cpu(),atol=atol,rtol=rtol)
        return {'passed':True,'device':str(device),'hidden_size':hidden_size,'atol':atol,'rtol':rtol,
                'max_abs_logit_diff':float((c_logits-d_logits.cpu()).abs().max().detach()),
                'value_max_abs_diff':float((c_v-d_v.cpu()).abs().max().detach()),
                'loss_cpu':float(cl.detach()),'loss_device':float(dl.detach()),'elapsed_seconds':time.monotonic()-start}
