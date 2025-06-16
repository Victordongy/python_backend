# flake8: noqa
import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """BLS model that sequentially invokes a generation and a classification
    model. The generated title is returned only if the classification
    score is below a configurable threshold."""

    def initialize(self, args):
        """Initialize the model and parse configuration."""
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})
        self.threshold = float(params.get("threshold", {}).get("string_value", 0.5))
        self.gen_model = params.get("generation_model", {}).get(
            "string_value", "tensorrt_llm_qwen0.5b"
        )
        self.cls_model = params.get("classification_model", {}).get(
            "string_value", "bert_small_trt"
        )

        title_config = pb_utils.get_output_config_by_name(self.model_config, "TITLE")
        self.title_dtype = pb_utils.triton_string_to_numpy(title_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT")

            # Call the generation model
            gen_request = pb_utils.InferenceRequest(
                model_name=self.gen_model,
                requested_output_names=["GENERATED_TITLE"],
                inputs=[prompt],
            )
            gen_response = gen_request.exec()
            if gen_response.has_error():
                raise pb_utils.TritonModelException(gen_response.error().message())

            generated_title = pb_utils.get_output_tensor_by_name(
                gen_response, "GENERATED_TITLE"
            )

            # Pass generated title to the classification model
            text_tensor = pb_utils.Tensor("TEXT", generated_title.as_numpy())
            class_request = pb_utils.InferenceRequest(
                model_name=self.cls_model,
                requested_output_names=["SCORE"],
                inputs=[text_tensor],
            )
            class_response = class_request.exec()
            if class_response.has_error():
                raise pb_utils.TritonModelException(class_response.error().message())

            score_tensor = pb_utils.get_output_tensor_by_name(class_response, "SCORE")
            score = float(score_tensor.as_numpy()[0])

            if score < self.threshold:
                out_title = pb_utils.Tensor(
                    "TITLE", generated_title.as_numpy().astype(self.title_dtype)
                )
            else:
                out_title = pb_utils.Tensor(
                    "TITLE", np.array([b""], dtype=self.title_dtype)
                )

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_title]))

        return responses

    def finalize(self):
        print("Cleaning up...")
