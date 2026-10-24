import flow


def test_word_loop_keeps_first_copy():
    assert flow.strip_hallucinations(
        "they are safe, safe, safe, safe, safe, safe.") == "they are safe."


def test_three_repeats_left_alone():
    assert flow.strip_hallucinations("no no no, not that") == "no no no, not that"


def test_two_word_unit_loop():
    assert flow.strip_hallucinations(
        "and then, here again, here again, here again, here again.") == "and then, here again."


def test_clean_text_byte_identical():
    t = "I really really like it, okay?"
    assert flow.strip_hallucinations(t) is t or flow.strip_hallucinations(t) == t
