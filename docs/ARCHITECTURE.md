# Architecture

`env.py` is the independent NumPy environment: deterministic movement, merges,
legal-action masks, then a uniform empty-cell spawn (2 with 90%, 4 with 10%). A game
ends only when no legal action remains. Environments have independent seeded RNGs.

`model.py` encodes tile exponents as one-hot vectors. The packaged 832-wide model
has 977,605 parameters. Policy logits are masked before categorical softmax.

`rollout.py` collects synchronous fixed-policy batches and stores actual post-action
observations before resetting finished environments. `gae.py` bootstraps through
truncation but not true termination, and stops traces across episode boundaries.
Advantages are normalized once across the full batch; value targets are unchanged.

`ppo.py` performs clipped policy optimization with unclipped value MSE, entropy
regularization, gradient norm clipping, and exact full-rollout KL after each epoch.
The default coefficient choices mirror the archived experiment's numerical core.

`train.py` is a new portable packaging entry point around these unchanged core
functions. It requires an explicit update/time bound and a fresh output directory.
It has no access to operational campaign paths. It does not resume historical
experiments or automatically load bundled inference models.

`inference.py` loads hash-checked tensor-only exports; `evaluate.py` isolates RNGs
and model mode, records natural terminal states, and can emit self-contained HTML
replays. Partial episodes are labelled and excluded from complete-score summaries.

`web/archive-base3/` plays fixed historical JSON only. Its timeline does not run
model inference. The standard-library server exposes only the web directory and
binds localhost.
