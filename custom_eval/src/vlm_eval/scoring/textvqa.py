from __future__ import annotations

import re

from vlm_eval.schemas import EvalSample
from vlm_eval.scoring.base import ScoreResult

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "isnt": "isn't", "itd": "it'd", "itll": "it'll", "maam": "ma'am",
    "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "oclock": "o'clock",
    "oughtnt": "oughtn't", "shant": "shan't", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodyd": "somebody'd", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
    "hed've": "he'd've", "he'dve": "he'd've",
    "itd've": "it'd've", "it'dve": "it'd've",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "notve": "not've", "shed've": "she'd've", "she'dve": "she'd've",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "thered've": "there'd've", "there'dve": "there'd've",
    "theyd've": "they'd've", "they'dve": "they'd've",
    "wed've": "we'd've", "we'dve": "we'd've",
    "whod've": "who'd've", "who'dve": "who'd've",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd've": "you'd've", "you'dve": "you'd've",
}
_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")


def normalize_textvqa_answer(answer: str) -> str:
    """Normalize an answer like MMF's EvalAIAnswerProcessor."""
    text = answer.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
    text = text.replace("\n", " ").replace("\t", " ")
    output = text
    for punctuation in _PUNCTUATION:
        if (
            f"{punctuation} " in text
            or f" {punctuation}" in text
            or _COMMA_STRIP.search(text) is not None
        ):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    output = _PERIOD_STRIP.sub("", output)

    words: list[str] = []
    for word in output.lower().split():
        word = _NUMBER_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


class TextVQAScorer:
    """Official VQA soft accuracy used by TextVQA validation evaluation."""

    def score(self, sample: EvalSample, prediction: str) -> ScoreResult:
        normalized_prediction = normalize_textvqa_answer(prediction)
        answers = [normalize_textvqa_answer(answer) for answer in sample.answers]
        if not answers:
            raise ValueError("TextVQA scoring requires ground-truth answers (use validation, not test)")

        per_annotator: list[float] = []
        indexed_answers = list(enumerate(answers))
        for current in indexed_answers:
            other_answers = [item for item in indexed_answers if item != current]
            matches = sum(answer == normalized_prediction for _, answer in other_answers)
            per_annotator.append(min(1.0, matches / 3.0))

        return ScoreResult(
            normalized_prediction=normalized_prediction,
            score=sum(per_annotator) / len(per_annotator),
        )
