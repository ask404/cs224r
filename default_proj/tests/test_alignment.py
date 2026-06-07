"""M0: token alignment / position shift — the silent off-by-one guard."""
from ziprc import alignment


def test_make_label_positions():
    prompt = [10, 11, 12, 13, 14]      # len 5
    output = [20, 21, 22, 23]          # len 4
    input_ids, lp = alignment.make_label_positions(prompt, output)
    assert input_ids == prompt + output
    assert lp == [5, 6, 7, 8]          # response token positions


def test_train_time_align_ttc_invariants():
    prompt = [10, 11, 12, 13, 14]
    output = [20, 21, 22, 23]
    input_ids, lp = alignment.make_label_positions(prompt, output)
    ids, pos, ttc = alignment.train_time_align(input_ids, lp, max_length=4096)

    assert ids == input_ids[:-1]                 # last token dropped
    assert pos == [4, 5, 6, 7]                   # shifted by -1
    assert ttc == [3, 2, 1, 0]                   # remaining tokens, 0 at last prefix
    assert ttc[-1] == 0                          # final prefix has nothing left
    assert all(ttc[i] > ttc[i + 1] for i in range(len(ttc) - 1))  # strictly decreasing


def test_train_time_align_respects_max_length():
    prompt = list(range(5))
    output = list(range(100, 100 + 50))
    input_ids, lp = alignment.make_label_positions(prompt, output)
    ids, pos, ttc = alignment.train_time_align(input_ids, lp, max_length=10)
    assert len(ids) == 10
    assert all(0 <= p < len(ids) for p in pos)   # no out-of-range positions
    assert all(t >= 0 for t in ttc)


def test_alignment_predicts_first_response_token():
    # The prefix BEFORE the first response token must be a labeled position,
    # i.e. the model state that *predicts* the first response token is supervised.
    prompt = [10, 11, 12, 13, 14]
    output = [20, 21, 22, 23]
    input_ids, lp = alignment.make_label_positions(prompt, output)
    ids, pos, ttc = alignment.train_time_align(input_ids, lp, max_length=4096)
    # position 4 (last prompt token) has hidden state predicting token at index 5
    # (= first response token), and is the first supervised position.
    assert pos[0] == len(prompt) - 1 == 4


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
