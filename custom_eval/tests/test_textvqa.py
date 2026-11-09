import json

import pytest

from vlm_eval import EvaluationRunner, JsonlStore, TextVQALoader, TextVQAScorer
from vlm_eval.models import DummyAdapter
from vlm_eval.scoring import normalize_textvqa_answer


def annotation() -> dict:
    return {
        "dataset_name": "textvqa",
        "dataset_type": "val",
        "data": [
            {
                "question_id": 101,
                "question": "How many cats are shown?",
                "image_id": "image_a",
                "set_name": "val",
                "image_width": 640,
                "image_height": 480,
                "answers": ["two"] * 3 + ["three"] * 7,
            },
            {
                "question_id": 102,
                "question": "What word is visible?",
                "image_id": "image_b",
                "set_name": "val",
                "answers": ["hello"] * 10,
            },
        ],
    }


def test_loader_reads_official_json_shape(tmp_path) -> None:
    annotation_path = tmp_path / "TextVQA_val.json"
    annotation_path.write_text(json.dumps(annotation()), encoding="utf-8")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "image_a.jpg").touch()

    samples = list(TextVQALoader(image_dir=image_dir).load_json(annotation_path, limit=1))

    assert len(samples) == 1
    assert samples[0].benchmark == "textvqa"
    assert samples[0].sample_id == "101"
    assert samples[0].media[0] == image_dir / "image_a.jpg"
    assert len(samples[0].answers) == 10


def test_loader_accepts_huggingface_style_image_object() -> None:
    image_object = object()
    row = annotation()["data"][0] | {"image": image_object}

    sample = next(TextVQALoader().from_rows([row]))

    assert sample.media == [image_object]
    assert sample.media_refs == ["image_a"]


def test_textvqa_normalization_matches_evalai_rules() -> None:
    assert normalize_textvqa_answer("The TWO, cats!") == "2 cats"
    assert normalize_textvqa_answer("Dont") == "don't"


@pytest.mark.parametrize(
    ("matching_answers", "expected"),
    [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)],
)
def test_official_soft_vqa_score(matching_answers, expected) -> None:
    row = annotation()["data"][0] | {
        "answers": ["yes"] * matching_answers + ["no"] * (10 - matching_answers)
    }
    sample = next(TextVQALoader().from_rows([row]))

    result = TextVQAScorer().score(sample, "yes")

    assert result.score == pytest.approx(expected)


def test_textvqa_runs_through_complete_pipeline(tmp_path) -> None:
    samples = list(TextVQALoader().from_rows(annotation()["data"]))
    store = JsonlStore(tmp_path / "textvqa" / "predictions.jsonl")
    runner = EvaluationRunner(
        run_id="textvqa-dummy",
        model=DummyAdapter(responses={"101": "two", "102": "hello"}),
        scorer=TextVQAScorer(),
        store=store,
    )

    summary = runner.run(samples)

    assert summary.processed == 2
    assert summary.mean_score == pytest.approx(0.95)
    assert len(store) == 2
