from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 1
    hidden_size: int = 256
    num_envs: int = 128
    rollout_steps: int = 128
    total_iterations: int = 61
    minibatch_size: int = 1024
    epochs: int = 4
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = .999
    gae_lambda: float = .95
    reward_scale: float = 1024.
    clip_epsilon: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .01
    max_grad_norm: float = .5
    target_kl: float = .02
    threads: int = 1
    deterministic: bool = True
    eval_every: int = 10
    light_eval_episodes: int = 10
    validation_episodes: int = 50
    test_episodes: int = 100
    test_split_id: int = 2
    max_episode_steps: int | None = None
    save_reserve_seconds: float = 30.
    eval_reserve_seconds: float = 180.

    def validate(self):
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError('hidden_size must be a positive integer')
        positive = ('num_envs', 'rollout_steps', 'total_iterations', 'minibatch_size', 'epochs',
                    'learning_rate', 'adam_eps', 'reward_scale', 'threads', 'max_grad_norm', 'target_kl')
        for field in positive:
            if getattr(self, field) <= 0:
                raise ValueError(f'{field} must be positive')
        if self.seed < 0 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid seed/gamma/lambda')
        if self.batch_size % self.minibatch_size:
            raise ValueError('batch_size must be divisible by minibatch_size')
        if not 0 < self.clip_epsilon < 1:
            raise ValueError('clip epsilon must be in (0,1)')
        if min(self.eval_every, self.light_eval_episodes, self.validation_episodes, self.test_episodes) < 0:
            raise ValueError('evaluation sizes must be nonnegative')
        if not isinstance(self.test_split_id,int) or isinstance(self.test_split_id,bool) or self.test_split_id < 2:
            raise ValueError('test split must be an integer >=2; 0/1 are training/validation')
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError('invalid episode limit')
        if min(self.save_reserve_seconds, self.eval_reserve_seconds) < 0:
            raise ValueError('invalid budget reserves')
        return self

    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')
