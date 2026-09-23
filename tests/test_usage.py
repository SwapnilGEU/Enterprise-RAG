"""Token accounting — pure functions, no Ollama needed."""

from types import SimpleNamespace

from src.usage import llm_usage, total_tokens_per_sec


def reply(prompt=None, completion=None, eval_ns=0, via_usage_metadata=True):
    """A stand-in for a ChatOllama AIMessage."""
    if via_usage_metadata:
        return SimpleNamespace(
            usage_metadata={"input_tokens": prompt, "output_tokens": completion},
            response_metadata={"eval_duration": eval_ns},
        )
    return SimpleNamespace(
        usage_metadata=None,
        response_metadata={"prompt_eval_count": prompt, "eval_count": completion,
                           "eval_duration": eval_ns},
    )


def test_single_reply():
    usage = llm_usage([reply(300, 50, eval_ns=2_000_000_000)])
    assert usage == {"llm_calls": 1, "prompt_tokens": 300, "completion_tokens": 50,
                     "total_tokens": 350, "tokens_per_sec": 25.0}


def test_sums_every_model_turn_and_skips_non_llm_messages():
    human = SimpleNamespace(content="hi")                 # no usage at all
    usage = llm_usage([human, reply(100, 10, 1_000_000_000), reply(200, 30, 1_000_000_000)])
    assert usage["llm_calls"] == 2
    assert usage["total_tokens"] == 340
    assert usage["tokens_per_sec"] == 20.0                # 40 tokens / 2 s


def test_falls_back_to_raw_ollama_fields():
    usage = llm_usage([reply(120, 8, 400_000_000, via_usage_metadata=False)])
    assert usage["total_tokens"] == 128
    assert usage["tokens_per_sec"] == 20.0


def test_no_duration_means_no_rate_not_a_crash():
    assert llm_usage([reply(10, 5, eval_ns=0)])["tokens_per_sec"] is None
    assert llm_usage([])["total_tokens"] == 0


def test_total_tokens_per_sec():
    assert total_tokens_per_sec(500, 2.0) == 250.0
    assert total_tokens_per_sec(0, 2.0) is None
    assert total_tokens_per_sec(500, 0) is None
