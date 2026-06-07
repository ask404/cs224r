"""M0: grid binning math — index<->state roundtrip, edges, marginals."""
import numpy as np

from ziprc import grid
from ziprc.config import LENGTH_BINS_COUNTDOWN, REWARD_VALUES_STRUCTURED


def test_value_bin_edges_structured():
    edges = grid.value_bin_edges(REWARD_VALUES_STRUCTURED)  # [0.0, 0.1, 1.0]
    assert edges[0] == 0.0 and edges[-1] == 1.0
    # 0.0 -> state 0, 0.1 -> state 1, 1.0 -> state 2
    assert grid.reward_state_of(0.0, edges) == 0
    assert grid.reward_state_of(0.1, edges) == 1
    assert grid.reward_state_of(1.0, edges) == 2


def test_length_bin_monotone_and_endpoints():
    lb = LENGTH_BINS_COUNTDOWN
    assert grid.length_bin_of(0, lb) == 0          # ttc=0 -> smallest bin
    assert grid.length_bin_of(10_000, lb) == len(lb) - 2  # overflow -> last bin
    prev = -1
    for ttc in range(0, 1100, 7):
        b = grid.length_bin_of(ttc, lb)
        assert b >= prev  # non-decreasing in ttc
        prev = b


def test_bin_index_decompose_roundtrip():
    L = len(LENGTH_BINS_COUNTDOWN) - 1
    R = len(REWARD_VALUES_STRUCTURED)
    seen = set()
    for rs in range(R):
        for lb in range(L):
            idx = lb + rs * L
            assert grid.decompose(idx, L) == (rs, lb)
            seen.add(idx)
    assert seen == set(range(R * L))  # every cell hit exactly once, no overlap


def test_marginal_and_expected_value():
    L = len(LENGTH_BINS_COUNTDOWN) - 1
    R = len(REWARD_VALUES_STRUCTURED)
    # one-hot at (reward_state=2 'correct', length_bin=0)
    probs = np.zeros(R * L)
    probs[0 + 2 * L] = 1.0
    pr = grid.marginal_reward(probs, R, L)
    assert pr.shape == (R,)
    assert np.isclose(pr[2], 1.0) and np.isclose(pr[:2].sum(), 0.0)
    ev = grid.expected_value(probs, REWARD_VALUES_STRUCTURED, LENGTH_BINS_COUNTDOWN)
    assert np.isclose(ev, 1.0)  # equals reward_values[2]

    # uniform over reward states (length fixed) -> mean of reward_values
    probs2 = np.zeros(R * L)
    for rs in range(R):
        probs2[0 + rs * L] = 1.0 / R
    ev2 = grid.expected_value(probs2, REWARD_VALUES_STRUCTURED, LENGTH_BINS_COUNTDOWN)
    assert np.isclose(ev2, np.mean(REWARD_VALUES_STRUCTURED))


def test_bin_index_matches_manual():
    # correct + ttc=0  -> reward_state 2, length_bin 0 -> idx = 0 + 2*L
    L = len(LENGTH_BINS_COUNTDOWN) - 1
    idx = grid.bin_index(0, 1.0, REWARD_VALUES_STRUCTURED, LENGTH_BINS_COUNTDOWN)
    assert idx == 2 * L


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
