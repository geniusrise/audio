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

from typing import Any, Dict, Optional, List, Union
import json
import os
import glob
import torchaudio
import uuid
import torch
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_audio.base import AudioBulk
from transformers import AutoModelForCTC, AutoProcessor


class SpeechToTextBulk(AudioBulk):
    """
    SpeechToTextBulk is designed for bulk processing of speech-to-text tasks. It efficiently processes large datasets of audio files,
    converting speech to text using Hugging Face's Wav2Vec2 models.

    Attributes:
        model (AutoModelForCTC): The speech-to-text model.
        processor (AutoProcessor): The processor to prepare input audio data for the model.

    Methods:
        transcribe_batch(audio_files: List[str], **kwargs: Any) -> List[str]:
            Transcribes a batch of audio files to text.
    """

    model: AutoModelForCTC
    processor: AutoProcessor

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the SpeechToTextBulk with configurations for speech-to-text processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)

    def transcribe(
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
        batch_size: int = 8,
        notification_email: Optional[str] = None,
        input_sampling_rate: int = 48_000,
        model_sampling_rate: int = 16_000,
        **kwargs: Any,
    ) -> List[str]:
        """
        Transcribes a batch of audio files to text using the speech-to-text model.

        Args:
            model_name (str): Name or path of the model.
            model_class (str): Class name of the model (default "AutoModelForSequenceClassification").
            processor_class (str): Class name of the processor (default "Autoprocessor").
            use_cuda (bool): Whether to use CUDA for model inference (default False).
            precision (str): Precision for model computation (default "float").
            quantization (int): Level of quantization for optimizing model size and speed (default 0).
            device_map (str | Dict | None): Specific device to use for computation (default "auto").
            max_memory (Dict): Maximum memory configuration for devices.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            batch_size (int): Number of classifications to process simultaneously (default 8).
            **kwargs: Arbitrary keyword arguments for model and generation configurations.
        """
        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            processor_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            processor_name = model_name
        else:
            model_revision = None
            processor_revision = None
            processor_name = model_name

        self.model_name = model_name
        self.processor_name = processor_name
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.batch_size = batch_size
        self.notification_email = notification_email
        self.compile = compile

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

        generation_args = {k.replace("generation_", ""): v for k, v in kwargs.items() if "generation_" in k}
        self.generation_args = generation_args

        processor_args = {k.replace("processor_", ""): v for k, v in kwargs.items() if "processor_" in k}
        self.processor_args = processor_args

        self.model, self.processor = self.load_models(
            model_name=self.model_name,
            processor_name=self.processor_name,
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            model_class=self.model_class,
            processor_class=self.processor_class,
            use_cuda=self.use_cuda,
            precision=self.precision,
            quantization=self.quantization,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            compile=self.compile,
            **self.model_args,
        )

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        audio_files = []
        for filename in glob.glob(f"{dataset_path}/**/*", recursive=True):
            extension = os.path.splitext(filename)[1]
            if extension.lower() not in [".wav", ".mp3", ".flac", ".ogg"]:
                continue
            filepath = os.path.join(dataset_path, filename)
            audio_files.append(filepath)

        # process batchwise
        transcriptions = []
        for i in range(0, len(audio_files), self.batch_size):
            batch = audio_files[i : i + self.batch_size]

            input_values = []
            for audio_file in batch:
                # Load and preprocess audio
                audio_input, sampling_rate = self._load_audio(audio_file)
                audio_input = torchaudio.functional.resample(
                    audio_input, orig_freq=input_sampling_rate, new_freq=model_sampling_rate
                )
                input_values.append(
                    self.processor(
                        audio_input, sampling_rate=sampling_rate, return_tensors="pt", **processor_args
                    ).input_values
                )

            # Perform inference
            with torch.no_grad():
                logits = self.model.generate(torch.stack(input_values, dim=0), **generation_args).logits

            # Decode the model output
            predicted_ids = torch.argmax(logits, dim=-1)
            for pid in predicted_ids:
                transcription = self.processor.decode(pid)
                transcriptions.append(transcription)

            self._save_transcriptions(
                transcriptions=transcriptions, filenames=batch, batch_idx=i, output_path=output_path
            )

        return transcriptions

    def _load_audio(self, file_path: str) -> torchaudio.tensor:
        """
            Loads an audio file and preprocesses it for the speech-to-text model.
        Args:
            file_path (str): Path to the audio file.

        Returns:
            Tuple[torch.Tensor, int]: Tuple containing the audio waveform and the sampling rate.
        """
        waveform, sampling_rate = torchaudio.load(file_path)
        return waveform, sampling_rate

    def _save_transcriptions(self, filenames: List[str], transcriptions: List[str], batch_idx: int, output_path: str):
        """
        Saves the transcriptions to the specified output folder.

        Args:
            filenames (List[str]): List of filenames of the transcribed audio files.
            transcriptions (List[str]): List of transcribed texts.
            output_path (str): Path to the output folder.
            batch_idx (int): Index of the current batch (for naming files).
        """
        data_to_save = [
            {"input": filename, "prediction": transcription}
            for filename, transcription in zip(filenames, transcriptions)
        ]

        with open(
            os.path.join(output_path, f"predictions-{batch_idx}-{str(uuid.uuid4())}.json"),
            "w",
        ) as f:
            json.dump(data_to_save, f)
