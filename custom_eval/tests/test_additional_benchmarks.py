import pytest

from vlm_eval import ChartQALoader, ChartQAScorer, MMELoader, MMEScorer, OCRBenchLoader, OCRBenchScorer
from vlm_eval.scoring import aggregate_mme


def test_chartqa_loader_and_relaxed_numeric_score() -> None:
    sample = next(ChartQALoader().from_rows([{
        "type": "human_test", "question": "Value?", "answer": "100", "image": object()
    }]))
    assert sample.metadata["question_type"] == "human_test"
    assert ChartQAScorer().score(sample, "104").score == 1.0
    assert ChartQAScorer().score(sample, "106").score == 0.0


def test_ocrbench_loader_and_containment_score() -> None:
    sample = next(OCRBenchLoader().from_rows([{
        "dataset": "IIIT5K", "question_type": "Regular Text Recognition",
        "question": "Read it", "answer": ["CENTRE"], "image": object(),
    }]))
    assert OCRBenchScorer().score(sample, "The text is CENTRE.").score == 1.0


def test_ocrbench_math_expression_is_strict_except_spaces() -> None:
    sample = next(OCRBenchLoader().from_rows([{
        "dataset": "HME100K", "question_type": "Handwritten Mathematical Expression Recognition",
        "question": "Read formula", "answer": ["x + 1"], "image": object(),
    }]))
    assert OCRBenchScorer().score(sample, "x+1").score == 1.0
    assert OCRBenchScorer().score(sample, "formula x+1").score == 0.0


def test_mme_loader_yes_no_and_official_acc_plus_aggregation() -> None:
    rows = [
        {"question_id": "existence/a.png", "image": object(), "question": "Is it?", "answer": "Yes", "category": "existence"},
        {"question_id": "existence/a.png", "image": object(), "question": "Isn't it?", "answer": "No", "category": "existence"},
    ]
    samples = list(MMELoader().from_rows(rows))
    assert len({sample.sample_id for sample in samples}) == 2
    assert MMEScorer().score(samples[0], "Yes, it is.").score == 1.0
    records = [
        {"sample_id": "1", "score": 1.0, "error": None, "metadata": {"category": "existence", "image_id": "a"}},
        {"sample_id": "2", "score": 1.0, "error": None, "metadata": {"category": "existence", "image_id": "a"}},
        {"sample_id": "3", "score": 1.0, "error": None, "metadata": {"category": "existence", "image_id": "b"}},
        {"sample_id": "4", "score": 0.0, "error": None, "metadata": {"category": "existence", "image_id": "b"}},
    ]
    result = aggregate_mme(records)
    assert result["categories"]["existence"]["accuracy"] == pytest.approx(75.0)
    assert result["categories"]["existence"]["accuracy_plus"] == pytest.approx(50.0)
    assert result["perception_score"] == pytest.approx(125.0)


def test_mme_acc_plus_ignores_an_incomplete_question_pair() -> None:
    result = aggregate_mme([
        {"sample_id": "1", "score": 1.0, "error": None,
         "metadata": {"category": "code_reasoning", "image_id": "a"}},
    ])
    category = result["categories"]["code_reasoning"]
    assert category["accuracy"] == 100.0
    assert category["accuracy_plus"] == 0.0
    assert category["complete_pairs"] == 0
