# flake8: noqa
"""Latency benchmark for the sequential pipeline example.

The real pipeline uses TensorRT-LLM with the QWen-0.5B model and a
TensorRT-optimized BERT classifier. This test does not rely on those
large models. Instead it uses tiny scikit-learn models to simulate
generation and classification so that the latency of calling the two
models through the pipeline can be compared against calling them
separately.
"""

import time
from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


PROMPT = "this is a test prompt"
ITERATIONS = 10
THRESHOLD = 0.5


# Minimal training data for the naive Bayes classifier. 1 denotes a "bad" title
# while 0 indicates acceptable content.
_TRAIN_TEXTS: List[str] = [
    "bad example",
    "unwanted title",
    "excellent",
    "very good",
]
_TRAIN_LABELS = [1, 1, 0, 0]

_VECTORIZER = CountVectorizer()
_TRAIN_X = _VECTORIZER.fit_transform(_TRAIN_TEXTS)
_CLASSIFIER = MultinomialNB().fit(_TRAIN_X, _TRAIN_LABELS)


def generation_model(prompt: str) -> str:
    """Very small generation model that returns the prompt capitalized."""

    return prompt.capitalize()


def classification_model(title: str) -> float:
    """Return the probability of the title being unacceptable."""

    x = _VECTORIZER.transform([title])
    # Probability of label 1 (bad title)
    return float(_CLASSIFIER.predict_proba(x)[0][1])


def run_pipeline() -> float:
    """Run the generation followed by classification and threshold check."""

    start = time.perf_counter()
    title = generation_model(PROMPT)
    score = classification_model(title)
    _ = title if score < THRESHOLD else ""
    return time.perf_counter() - start


def run_manual() -> float:
    """Run generation and classification separately without threshold logic."""

    start = time.perf_counter()
    title = generation_model(PROMPT)
    classification_model(title)
    return time.perf_counter() - start


def main():
    pipe_times = []
    manual_times = []
    for _ in range(ITERATIONS):
        pipe_times.append(run_pipeline())
        manual_times.append(run_manual())

    print("Average latency with BLS: {:.4f}s".format(sum(pipe_times) / ITERATIONS))
    print("Average latency without BLS: {:.4f}s".format(sum(manual_times) / ITERATIONS))


if __name__ == "__main__":
    main()
