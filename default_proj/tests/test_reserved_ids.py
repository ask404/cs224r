"""M0: reserved-logit slice is in-bounds and collision-free for Qwen2.5-0.5B."""
import json
import os

from ziprc import config


def test_distribution_token_id_is_qwen25_not_qwen3():
    assert config.DISTRIBUTION_TOKEN_ID == 151665
    assert config.DISTRIBUTION_TOKEN_ID != 151669  # the upstream Qwen3 id


def test_structured_slice_fits():
    nb = config.assert_slice_fits(config.REWARD_VALUES_STRUCTURED)
    assert nb == len(config.REWARD_VALUES_STRUCTURED) * (len(config.LENGTH_BINS_COUNTDOWN) - 1)
    assert config.DISTRIBUTION_TOKEN_ID + nb <= config.VOCAB_SIZE


def test_binary_slice_fits():
    config.assert_slice_fits(config.REWARD_VALUES_BINARY)


def test_collision_with_real_tokens_raises():
    bad_start = config.HIGHEST_REAL_TOKEN_ID  # 151664, a real token
    try:
        config.assert_slice_fits(config.REWARD_VALUES_STRUCTURED, start=bad_start)
    except ValueError:
        return
    raise AssertionError("expected ValueError for colliding start id")


def test_overflow_raises():
    try:
        config.assert_slice_fits(config.REWARD_VALUES_STRUCTURED, start=config.VOCAB_SIZE - 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overflowing slice")


def test_matches_local_checkpoint_if_present():
    """If the downloaded SFT checkpoint is here, assert our constants match it."""
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "downloaded_checkpoints",
        "sft_final_model", "config.json",
    )
    if not os.path.exists(cfg_path):
        return  # skip when checkpoint not available (e.g. on Modal)
    vocab = json.load(open(cfg_path))["vocab_size"]
    assert vocab == config.VOCAB_SIZE
    assert config.DISTRIBUTION_TOKEN_ID > config.HIGHEST_REAL_TOKEN_ID
    assert config.HIGHEST_REAL_TOKEN_ID < vocab


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
