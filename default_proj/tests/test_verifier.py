"""M0: three-outcome verifier on hand-crafted strings (heuristic backend)."""
from ziprc import verifier
from ziprc.config import COHERENT, CORRECT, INCOHERENT

GT = {"target": 98, "numbers": [44, 19, 35]}


def test_correct_is_correct():
    # (44 + 19) + 35 = 98, uses all three numbers exactly once
    r = "<think> 44+19=63, 63+35=98 </think><answer>(44 + 19) + 35</answer>"
    assert verifier.three_outcome_label(r, GT) == CORRECT
    assert verifier.binary_label(r, GT) == 1.0


def test_coherent_failure():
    # Real arithmetic with the allowed numbers, but wrong/incomplete final answer.
    r = "<think> 44 + 19 = 63, that's close </think><answer>44 + 19</answer>"
    out = verifier.three_outcome_label(r, GT)
    assert out == COHERENT
    assert verifier.binary_label(r, GT) == 0.0  # binary collapses it to 0


def test_incoherent_wrong_numbers():
    r = "<think> 7 * 8 = 56, 56 + 100 = 156 </think><answer>7 * 8</answer>"
    assert verifier.three_outcome_label(r, GT) == INCOHERENT


def test_incoherent_gibberish():
    r = "the the the answer is probably a banana <answer>banana</answer>"
    assert verifier.three_outcome_label(r, GT) == INCOHERENT


def test_no_answer_tag_but_real_math_is_coherent():
    # No <answer> => rule-based 0.0, but genuine arithmetic over allowed numbers.
    r = "<think> 44 * 35 = 1540, 1540 / 19 is not integer </think>"
    assert verifier.three_outcome_label(r, GT) == COHERENT


def test_heuristic_returns_bool():
    assert verifier.coherence_heuristic("44 - 19 = 25", GT) is True
    assert verifier.coherence_heuristic("hello world", GT) is False


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
