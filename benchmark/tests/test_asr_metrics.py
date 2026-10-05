# benchmark/tests/test_asr_metrics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.metrics.asr_metrics import normalise, wer, cer, edit_distance, wer_counts, cer_counts


def test_normalise_en_lowercase():
    assert normalise("Hello World!", "en") == "hello world"


def test_normalise_en_punctuation():
    assert normalise("It's a test, right?", "en") == "its a test right"


def test_normalise_id_same_as_en():
    assert normalise("Halo Dunia!", "id") == "halo dunia"


def test_normalise_zh_nfkc():
    # Full-width digits should be normalised
    result = normalise("你好，世界。", "zh")
    assert "，" not in result
    assert "。" not in result
    assert "你好" in result
    assert "世界" in result


def test_edit_distance_identical():
    assert edit_distance(list("abc"), list("abc")) == 0


def test_edit_distance_insertion():
    assert edit_distance(list("ab"), list("abc")) == 1


def test_edit_distance_deletion():
    assert edit_distance(list("abc"), list("ab")) == 1


def test_edit_distance_substitution():
    assert edit_distance(list("abc"), list("axc")) == 1


def test_wer_perfect():
    assert wer("hello world", "hello world", "en") == 0.0


def test_wer_one_word_error():
    # "hello earth" vs "hello world" → 1 substitution / 2 words = 0.5
    result = wer("hello earth", "hello world", "en")
    assert abs(result - 0.5) < 0.001


def test_wer_empty_hypothesis():
    # 3 deletions / 3 words = 1.0
    result = wer("", "hello brave world", "en")
    assert result == 1.0


def test_cer_perfect():
    assert cer("你好世界", "你好世界") == 0.0


def test_cer_one_char_error():
    # "你好界" vs "你好世界" → 1 deletion / 4 chars = 0.25
    result = cer("你好界", "你好世界")
    assert abs(result - 0.25) < 0.001


def test_normalise_apostrophe_side_independent():
    # "don't" must equal "dont" so a missing apostrophe is not counted as an error
    assert wer("dont stop", "don't stop", "en") == 0.0
    assert wer("don\u2019t stop", "don't stop", "en") == 0.0


def test_normalise_hyphen_becomes_space():
    assert normalise("well-known", "en") == "well known"


def test_normalise_keeps_accented_letters():
    assert normalise("Café!", "en") == "café"


def test_normalise_zh_strips_all_punctuation():
    assert normalise("院子，《门口》、不远处…就是。", "zh") == "院子门口不远处就是"


def test_wer_counts_values():
    # ref has 4 words; hyp has 1 substitution + 1 deletion = 2 edits
    assert wer_counts("the cat sat", "the dog sat down", "en") == (2, 4)


def test_cer_counts_ignores_spaces():
    assert cer_counts("你好 世界", "你好世界") == (0, 4)


def test_wer_can_exceed_one():
    assert wer("a b c d e f", "a", "en") == 5.0


def test_wer_both_empty_is_zero():
    assert wer("", "", "en") == 0.0


def test_wer_empty_reference_nonempty_hyp_is_one():
    assert wer("something", "", "en") == 1.0


def test_wer_is_case_and_punct_insensitive():
    assert wer("HELLO, WORLD!", "hello world", "en") == 0.0


def test_corpus_rate_differs_from_mean_of_rates():
    # file A: 1 error / 1 word = 1.0 ; file B: 0 errors / 9 words = 0.0
    e1, n1 = wer_counts("x", "a", "en")
    e2, n2 = wer_counts("a b c d e f g h i", "a b c d e f g h i", "en")
    assert (e1 + e2) / (n1 + n2) == 0.1       # corpus-level
    assert (1.0 + 0.0) / 2 == 0.5             # naive mean would be 5x too high
