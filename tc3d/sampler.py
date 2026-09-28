"""
Custom Metropolis rules for the toric code: a probability-weighted mixture of
rules (WeightedRule) and a cluster flip of one stabilizer's edges (MultiRule).
The sampler itself is assembled in `tc3d.builders.build_sampler`.
"""

import jax
import jax.numpy as jnp
import netket as nk
from typing import Any, Optional, Tuple

@nk.utils.struct.dataclass
class WeightedRule(nk.sampler.rules.MetropolisRule):
    """A Metropolis sampling rule that can be used to combine different rules acting
    on different subspaces of the same tensor-hilbert space. Thanks to Marc Machaczek and Filippo Vicentini for this input.
    """
    probabilities: jax.Array
    rules: Tuple[nk.sampler.rules.MetropolisRule, ...]

    def __post_init__(self):
        if not isinstance(self.probabilities, jax.Array):
            object.__setattr__(self, "probabilities", jnp.array(self.probabilities))

        if not isinstance(self.rules, (tuple, list)):
            raise TypeError(
                "The second argument (rules) must be a tuple of `MetropolisRule` "
                f"rules, but you have passed {type(self.rules)}."
            )

        if len(self.probabilities) != len(self.rules):
            raise ValueError(
                "Length mismatch between the probabilities and the rules: probabilities "
                f"has length {len(self.probabilities)} , rules has length {len(self.rules)}."
            )

    def init_state(
            self,
            sampler: "nk.sampler.MetropolisSampler",
            machine: Any,
            params: Any,
            key: Any,
    ) -> Optional[Any]:
        N = len(self.probabilities)
        keys = jax.random.split(key, N)
        return tuple(
            self.rules[i].init_state(sampler, machine, params, keys[i])
            for i in range(N)
        )

    def reset(
            self,
            sampler: "nk.sampler.MetropolisSampler",
            machine: Any,
            params: Any,
            sampler_state: "nk.sampler.SamplerState",
    ) -> Optional[Any]:
        rule_states = []
        for i in range(len(self.probabilities)):
            # construct temporary sampler and rule state with correct sub-hilbert and
            # sampler-state objects.
            _state = sampler_state.replace(rule_state=sampler_state.rule_state[i])
            rule_states.append(self.rules[i].reset(sampler, machine, params, _state))
        return tuple(rule_states)

    def transition(self, sampler, machine, parameters, state, key, sigma):
        N = len(self.probabilities)
        keys = jax.random.split(key, N + 1)

        sigmaps = []
        log_prob_corrs = []
        for i in range(N):
            # construct temporary rule state with correct sampler-state objects
            _state = state.replace(rule_state=state.rule_state[i])

            sigmaps_i, log_prob_corr_i = self.rules[i].transition(sampler, machine, parameters, _state, keys[i], sigma)

            sigmaps.append(sigmaps_i)
            log_prob_corrs.append(log_prob_corr_i)

        indices = jax.random.choice(keys[-1], N, shape=(sampler.n_chains_per_rank,), p=self.probabilities)

        batch_select = jax.vmap(lambda s, idx: s[idx], in_axes=(1, 0), out_axes=0)
        sigmap = batch_select(jnp.stack(sigmaps), indices)  # sigmaps has dim (N, n_chains_per_rank, n_sites)

        # if not all log_prob_corr are 0, convert the Nones to 0s
        if any(x is not None for x in log_prob_corrs):
            log_prob_corrs = jnp.stack([x if x is not None else 0 for x in log_prob_corrs])
            log_prob_corr = batch_select(log_prob_corrs, indices)
        else:
            log_prob_corr = None

        return sigmap, log_prob_corr

    def __repr__(self):
        return f"WeightedRule(probabilities={self.probabilities}, rules={self.rules})"


@nk.utils.struct.dataclass
class MultiRule(nk.sampler.rules.MetropolisRule):
    """
    Updates/flips multiple spins according to update_clusters. One of the clusters provided is chosen at random,
    then all spins within that cluster are updated. Thanks to Marc Machaczek for this input.
    """
    update_clusters: jax.Array  # hashable array required? no bc not used as staticarg, but dynamicarg instead

    def transition(self, sampler, machine, parameters, state, key, sigmas):
        # Deduce the number of possible clusters to be updated
        n_clusters = self.update_clusters.shape[0]

        # Deduce the number of MCMC chains from input shape
        n_chains = sigmas.shape[0]

        # Split the rng key into 2: one for each random operation
        key_indx, key_flip = jax.random.split(key, 2)

        # Pick random cluster index on every chain. maxval is EXCLUSIVE, so it
        # must be n_clusters (not n_clusters-1) or the last vertex cluster can
        # never be flipped — losing one star move from the ergodic set.
        indxs = jax.random.randint(key_indx, shape=(n_chains, 1), minval=0, maxval=n_clusters)

        @jax.vmap
        def flip(sigma, cluster):
            return sigma.at[cluster].set(-sigma.at[cluster].get())

        sigmap = flip(sigmas, self.update_clusters[indxs])        # flip those clusters

        return sigmap, None  # second argument for potential correcting factor of L (not present for this rule)
