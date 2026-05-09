"""Verify Synapse produces correct LTP/LTD weight changes for the four
combinations of pairing direction (causal/acausal) x dopamine sign (reward/
punishment). Uses params from my_ros2_bringup/config/params.yaml.
"""

from python_snn_node.LIF_SNN_network import Synapse


PARAMS = dict(
    lr_shift=6,
    t_pre=3,
    t_post=3,
    tau_e_shift=4,
    dw_pos=5,
    dw_neg=5,
    w_min=16,
    w_max=254,
    mode='rstdp',
)
W_INIT = 128


def _fresh():
    return Synapse(w_init=W_INIT, **PARAMS)


def _causal_pair(syn):
    """Pre@t=0, post@t=1 -> eligibility positive (~+4 after first-tick decay)."""
    syn.update_eligibility(pre_spike=1, post_spike=0)
    syn.update_eligibility(pre_spike=0, post_spike=1)


def _acausal_pair(syn):
    """Post@t=0, pre@t=1 -> eligibility negative (~-4 after first-tick decay)."""
    syn.update_eligibility(pre_spike=0, post_spike=1)
    syn.update_eligibility(pre_spike=1, post_spike=0)


# With params dw_pos=dw_neg=5, tau_e_shift=4, lr_shift=6:
#   single pair gives |elig| ~= 4, so |dw| = (4 * |dop|) >> 6
# Need |dop| >= 16 to get |dw| >= 1. Real dopamine values from
# dopamine_reward_node are in [-6, +3], i.e. *below* this floor.
DOP_STRONG = 32


def test_causal_reward_is_LTP():
    syn = _fresh()
    _causal_pair(syn)
    assert syn.eligibility > 0, f"causal pairing should make eligibility positive, got {syn.eligibility}"
    syn.apply_reward(dopamine=+DOP_STRONG)
    assert syn.weight > W_INIT, f"causal+reward should be LTP, weight {W_INIT} -> {syn.weight}"


def test_causal_punishment_is_LTD():
    syn = _fresh()
    _causal_pair(syn)
    syn.apply_reward(dopamine=-DOP_STRONG)
    assert syn.weight < W_INIT, f"causal+punishment should be LTD, weight {W_INIT} -> {syn.weight}"


def test_acausal_reward_is_LTD():
    syn = _fresh()
    _acausal_pair(syn)
    assert syn.eligibility < 0, f"acausal pairing should make eligibility negative, got {syn.eligibility}"
    syn.apply_reward(dopamine=+DOP_STRONG)
    assert syn.weight < W_INIT, f"acausal+reward should be LTD, weight {W_INIT} -> {syn.weight}"


def test_acausal_punishment_is_LTP():
    """Standard symmetric R-STDP: punishment of an acausal trace strengthens.
    If you intend an asymmetric rule, this test is the one to flip.
    """
    syn = _fresh()
    _acausal_pair(syn)
    syn.apply_reward(dopamine=-DOP_STRONG)
    assert syn.weight > W_INIT, f"acausal+punishment should be LTP, weight {W_INIT} -> {syn.weight}"


def test_real_dopamine_range_is_below_truncation_floor():
    """Sanity check: the largest realistic dopamine value (+3 reward, -6 punish)
    applied to a single causal pairing produces a weight change of 0 due to
    integer truncation. This is informational, not necessarily a bug — it means
    learning requires either repeated pairings within a short window, or
    stronger dopamine."""
    syn = _fresh()
    _causal_pair(syn)
    elig = syn.eligibility
    syn.apply_reward(dopamine=+3)
    assert syn.weight == W_INIT, (
        f"Expected zero-change due to truncation, got {syn.weight - W_INIT} "
        f"(elig={elig}, dop=+3, lr_shift={PARAMS['lr_shift']})"
    )


def test_zero_dopamine_is_noop():
    syn = _fresh()
    _causal_pair(syn)
    elig_before = syn.eligibility
    syn.apply_reward(dopamine=0)
    assert syn.weight == W_INIT
    assert syn.eligibility == elig_before


def test_weight_clamps():
    syn = Synapse(w_init=PARAMS['w_max'], **PARAMS)
    _causal_pair(syn)
    syn.apply_reward(dopamine=+50)
    assert syn.weight == PARAMS['w_max']

    syn = Synapse(w_init=PARAMS['w_min'], **PARAMS)
    _causal_pair(syn)
    syn.apply_reward(dopamine=-50)
    assert syn.weight == PARAMS['w_min']


if __name__ == '__main__':
    tests = [
        test_causal_reward_is_LTP,
        test_causal_punishment_is_LTD,
        test_acausal_reward_is_LTD,
        test_acausal_punishment_is_LTP,
        test_zero_dopamine_is_noop,
        test_weight_clamps,
        test_real_dopamine_range_is_below_truncation_floor,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
