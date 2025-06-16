# Generation and Classification Pipeline Example

This example demonstrates how to use Business Logic Scripting (BLS) to
run two models sequentially. The first model generates a title from a
prompt using **TensorRT‑LLM** with the QWen‑0.5B model. The second model
scores that title using a small **BERT** model accelerated with
TensorRT. The BLS model returns the title only when the classification
score is below a configurable threshold.

See the [TensorRT](https://github.com/NVIDIA/TensorRT) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) repositories for more information on these frameworks.
The code in `pipeline_model.py` sends an inference request to the
generation model (default `tensorrt_llm_qwen0.5b`) and then sends the
generated text to the classification model (default `bert_small_trt`).
If the score returned by the classification model is greater than or
equal to the threshold, the final output is an empty string.

Both downstream model names and the threshold can be adjusted in the
model configuration.

## Measuring Pipeline Latency

Run [test_latency.py](test_latency.py) to compare the average latency of
using the BLS pipeline versus issuing separate requests to the
generation and classification models. The test uses tiny scikit‑learn
models so it can run without the TensorRT models installed:

```bash
python3 examples/bls_seq/test_latency.py
```

