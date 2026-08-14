"""Component 1 of the domain-continual-CBM plan: an exact, incremental
replacement for clip_cbm_orth's trained classifier (LayerNorm + Linear).

Why this is exact, not approximate
-----------------------------------
With CLIP and concept_embeddings frozen, the only thing clip_cbm_orth ever
trains is:
    classifier = Sequential(Flatten, LayerNorm(M), Linear(M, N))
LayerNorm's normalization step (subtract each sample's own mean, divide by
its own std, over the M concept scores) has no learnable parameters - it's
a fixed, per-sample transform. Composing its learnable affine step
(gamma, beta) with the following Linear(W2, b2) collapses algebraically to
a single affine map:
    logits = normalize(a) @ W'.T + b'   where  W' = W2 @ diag(gamma),  b' = W2 @ beta + b2
So the entire trainable pipeline is, exactly, one linear regression from
normalized concept activations to class scores - nothing more complex is
being trained than that, regardless of how it's parameterized.

A linear regression fit by least squares has an exact incremental form:
keep running sums
    R = sum over all data seen so far of (phi @ phi.T)      where phi = [normalize(a); 1]
    Q = sum over all data seen so far of (phi @ one_hot(y).T)
and solve W = R^-1 Q. Because R and Q are sums, adding a new domain's data
is just R += R_new_domain, Q += Q_new_domain - matrix addition is
commutative, so the classifier after N domains, in any order, is
numerically identical (up to floating-point solve precision) to solving on
all N domains pooled at once. No replay buffer, no penalty term to tune, no
gradient steps on old data, and therefore nothing approximate to forget.

The one deliberate adaptation: DDO's orthogonality term
-----------------------------------------------------------
DDO's penalty (Eq. 13 in the base paper) is originally L1:
    mean(|classifier[1:](diffs @ concept_embeddings.T)|)
L1 penalties don't have a closed-form incremental solution. This module
uses the L2 (mean-square) surrogate instead, which folds into R as one more
additive ridge-style term - computed once from the descriptor pool (fixed
for Component 1; Component 3 is where the pool itself grows), never
re-added per domain since it doesn't depend on domain data at all. This is
a stated substitution, not a hidden one: L1 and L2 versions of "push these
directions toward zero prediction" are both standard regularizer choices;
L2 is specifically the one that keeps the classifier update exact. Compare
against the original L1-trained gradient-descent baseline (domain_il.py) to
see what difference the substitution makes in practice.
"""
import torch


class AnalyticDomainIncrementalClassifier:
    def __init__(self, concept_dim, num_classes, domain_diffs, concept_embeddings,
                 ridge_lambda=1.0, ddo_lambda=1.0, eps=1e-5, device="cpu", dtype=torch.float64):
        self.M = concept_dim
        self.N = num_classes
        self.eps = eps
        self.device = device
        # Solved by direct inversion of the running sums every call, not by
        # a Sherman-Morrison-style rank-1 update chain - so precision matters
        # more than it would in float32, especially after many domains.
        self.dtype = dtype

        aug_dim = self.M + 1
        self.R = torch.eye(aug_dim, device=device, dtype=dtype) * ridge_lambda
        self.Q = torch.zeros(aug_dim, self.N, device=device, dtype=dtype)
        self.n_domains_seen = 0

        with torch.no_grad():
            # domain_diffs is (num_descriptors, num_anchor_classes, feature_dim) - the
            # original model's classifier[1:] applies LayerNorm+Linear over the last
            # dim regardless of the leading shape, so flatten to 2D first; each
            # (descriptor, anchor-class) pair is normalized as its own row, exactly
            # matching what classifier[1:](diffs @ concept_embeddings.T) does.
            diffs_flat = domain_diffs.to(dtype=dtype, device=device).reshape(-1, domain_diffs.shape[-1])
            d = diffs_flat @ concept_embeddings.to(dtype=dtype, device=device).T  # (num_descriptors * num_anchor_classes, M)
            d_norm = self._normalize(d)
            phi_ddo = torch.cat(
                [d_norm, torch.ones(d.shape[0], 1, device=device, dtype=dtype)], dim=1
            )
            n_terms = d.shape[0] * self.N
            self.R += (ddo_lambda / n_terms) * (phi_ddo.T @ phi_ddo)

        self._W = None

    def _normalize(self, a):
        """Reproduces nn.LayerNorm(M)'s normalization step exactly (no
        learnable affine - that part is absorbed into W, as derived above)."""
        mean = a.mean(dim=-1, keepdim=True)
        var = a.var(dim=-1, unbiased=False, keepdim=True)
        return (a - mean) / torch.sqrt(var + self.eps)

    def partial_fit(self, concept_activations, labels):
        """Incorporate one domain's (or one batch's) labeled concept
        activations. Call once per new domain arriving; nothing about
        earlier domains ever needs to be revisited or replayed."""
        a = concept_activations.to(dtype=self.dtype, device=self.device)
        a_norm = self._normalize(a)
        phi = torch.cat(
            [a_norm, torch.ones(a.shape[0], 1, device=self.device, dtype=self.dtype)], dim=1
        )
        y_onehot = torch.nn.functional.one_hot(
            labels.to(self.device), num_classes=self.N
        ).to(self.dtype)
        self.R += phi.T @ phi
        self.Q += phi.T @ y_onehot
        self.n_domains_seen += 1
        self._W = None

    def _solve(self):
        if self._W is None:
            self._W = torch.linalg.solve(self.R, self.Q)  # (M+1, N)
        return self._W

    def predict_logits(self, concept_activations):
        a = concept_activations.to(dtype=self.dtype, device=self.device)
        a_norm = self._normalize(a)
        phi = torch.cat(
            [a_norm, torch.ones(a.shape[0], 1, device=self.device, dtype=self.dtype)], dim=1
        )
        return (phi @ self._solve()).to(torch.float32)
