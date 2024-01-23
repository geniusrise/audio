# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformers
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoProcessor, AutoConfig

from geniusrise_audio.base.communication import send_email


class AudioBulk(Bolt):
    """
    AudioBulk is a class designed for bulk processing of audio data using various audio models from Hugging Face.
    It focuses on audio generation and transformation tasks, supporting a range of models and configurations.

    Attributes:
        model (AutoModelForAudioClassification): The audio model for generation or transformation tasks.
        processor (AutoFeatureExtractor): The processor for preparing input data for the model.

    Args:
        input (BatchInput): Configuration and data inputs for the batch process.
        output (BatchOutput): Configurations for output data handling.
        state (State): State management for the Bolt.
        **kwargs: Arbitrary keyword arguments for extended configurations.

    Methods:
        audio(**kwargs: Any) -> Dict[str, Any]:
            Provides an API endpoint for audio processing functionality.
            Accepts various parameters for customizing the audio processing tasks.

        process(audio_input: Union[str, bytes], **processing_params: Any) -> dict:
            Processes the audio input based on the provided parameters. Supports multiple processing methods.
    """

    model: AutoModelForAudioClassification
    processor: AutoFeatureExtractor | AutoProcessor

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the AudioBulk with configurations and sets up logging.
        Prepares the environment for audio processing tasks.

        Args:
            input (BatchInput): The input data configuration for the audio processing task.
            output (BatchOutput): The output data configuration for the results of the audio processing.
            state (State): The state configuration for the Bolt, managing its operational status.
            **kwargs: Additional keyword arguments for extended functionality and model configurations.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def load_models(
        self,
        model_name: str,
        processor_name: str,
        model_revision: Optional[str] = None,
        processor_revision: Optional[str] = None,
        model_class: str = "",
        processor_class: str = "AutoFeatureExtractor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: Union[str, Dict, None] = "auto",
        max_memory: Dict[int, str] = {0: "24GB"},
        torchscript: bool = False,
        compile: bool = True,
        **model_args: Any,
    ) -> Tuple[AutoModelForAudioClassification, AutoFeatureExtractor]:
        """
        Loads and configures the specified audio model and processor for audio processing.

        Args:
            model_name (str): Name or path of the audio model to load.
            processor_name (str): Name or path of the processor to load.
            model_revision (Optional[str]): Specific model revision to load (e.g., commit hash).
            processor_revision (Optional[str]): Specific processor revision to load.
            model_class (str): Class of the model to be loaded.
            processor_class (str): Class of the processor to be loaded.
            use_cuda (bool): Flag to use CUDA for GPU acceleration.
            precision (str): Desired precision for computations ("float32", "float16", etc.).
            quantization (int): Bit level for model quantization (0 for none, 8 for 8-bit).
            device_map (Union[str, Dict, None]): Specific device(s) for model operations.
            max_memory (Dict[int, str]): Maximum memory allocation for the model.
            torchscript (bool): Enable TorchScript for model optimization.
            compile (bool): Enable Torch JIT compilation.
            **model_args (Any): Additional arguments for model loading.

        Returns:
            Tuple[AutoModelForAudioClassification, AutoFeatureExtractor]: Loaded model and processor.
        """
        self.log.info(f"Loading audio model: {model_name}")

        # Determine torch dtype based on precision
        torch_dtype = self._get_torch_dtype(precision)

        # Configure device map for CUDA
        if use_cuda and not device_map:
            device_map = "auto"

        # Load the model and processor
        FeatureExtractorClass = getattr(transformers, processor_class)
        config = AutoConfig.from_pretrained(processor_name, revision=processor_revision)
        processor = FeatureExtractorClass.from_pretrained(processor_name, revision=processor_revision)

        ModelClass = getattr(transformers, model_class)
        if quantization == 8:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                max_memory=max_memory,
                load_in_8bit=True,
                config=config,
                **model_args,
            )
        elif quantization == 4:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                max_memory=max_memory,
                load_in_4bit=True,
                config=config,
                **model_args,
            )
        else:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch_dtype,
                max_memory=max_memory,
                config=config,
                **model_args,
            )

        model = model.to(device_map)
        if compile:
            model = torch.compile(model)

        # Set to evaluation mode for inference
        model.eval()

        self.log.debug("Audio model and processor loaded successfully.")
        return model, processor

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """
        Determines the torch dtype based on the specified precision.

        Args:
            precision (str): The desired precision for computations.

        Returns:
            torch.dtype: The corresponding torch dtype.

        Raises:
            ValueError: If an unsupported precision is specified.
        """
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float,
            "float64": torch.float64,
            "double": torch.double,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "half": torch.half,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "short": torch.short,
            "int32": torch.int32,
            "int": torch.int,
            "int64": torch.int64,
            "quint8": torch.quint8,
            "qint8": torch.qint8,
            "qint32": torch.qint32,
        }
        return dtype_map.get(precision, torch.float)

    def done(self):
        """
        Finalizes the AudioBulk processing. Sends notification email if configured.

        This method should be called after all audio processing tasks are complete.
        It handles any final steps such as sending notifications or cleaning up resources.
        """
        if self.notification_email:
            self.output.flush()
            send_email(recipient=self.notification_email, bucket_name=self.output.bucket, prefix=self.output.s3_folder)
