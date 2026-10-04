# benchmark/tests/test_asr_metrics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.metrics.asr_metrics import normalise, wer, cer, edit_distance


def test_normalise_en_lowercase():
    assert normalise("Hello World!", "en") == "hello world"


def test_normalise_en_punctuation():
    assert normalise("It's a test, right?", "en") == "it s a test right"


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
