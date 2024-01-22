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

import base64
import io
from pydub import AudioSegment
import cherrypy
import torch
import torchaudio
from typing import Tuple
from geniusrise import BatchInput, BatchOutput, State
from transformers import AutoModelForCTC, AutoProcessor

from geniusrise_audio.base import AudioAPI


class SpeechToTextAPI(AudioAPI):
    """
    SpeechToTextAPI is a subclass of AudioAPI specifically designed for speech-to-text models.
    It extends the functionality to handle speech-to-text processing using various ASR models.

    Attributes:
        model (AutoModelForCTC): The speech-to-text model.
        processor (AutoProcessor): The processor to prepare input audio data for the model.

    Methods:
        transcribe(audio_input: bytes) -> str:
            Transcribes the given audio input to text using the speech-to-text model.
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
        Initializes the SpeechToTextAPI with configurations for speech-to-text processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def transcribe(self):
        """
        API endpoint to transcribe the given audio input to text using the speech-to-text model.
        Expects a JSON input with 'audio_file' as a key containing the base64 encoded audio data.

        Returns:
            Dict[str, str]: A dictionary containing the transcribed text.
        """
        input_json = cherrypy.request.json
        audio_data = input_json.get("audio_file")
        # input_sampling_rate = input_json.get("input_sampling_rate", 48_000)
        model_sampling_rate = input_json.get("model_sampling_rate", 16_000)
        processor_args = input_json.get("processor_args", {})
        generate_args = input_json.get("generate_args", {})
        chunk_size = input_json.get("chunk_size", 0)
        overlap_size = input_json.get("overlap_size", 1600)

        # TODO: take temperature etc as args, also semantic_temperature
        # TODO: support voice presets

        if not audio_data:
            raise cherrypy.HTTPError(400, "No audio data provided.")

        # Convert base64 encoded data to bytes
        audio_input, input_sampling_rate = self.decode_audio(audio_data)
        audio_input = torchaudio.functional.resample(
            audio_input, orig_freq=input_sampling_rate, new_freq=model_sampling_rate
        )

        # Perform inference
        with torch.no_grad():
            if self.model.config.model_type == "whisper":
                transcription = self.process_whisper(
                    audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size, generate_args
                )
            elif self.model.config.model_type == "wav2vec2":
                transcription = self.process_wav2vec2(
                    audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size
                )

        return {"transcription": transcription}

    def process_whisper(
        self, audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size, generate_args
    ):
        """
        Process audio input with the Whisper model.
        """
        # Preprocess and transcribe
        input_values = self.processor(
            audio_input.squeeze(0),
            return_tensors="pt",
            sampling_rate=model_sampling_rate,
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            do_normalize=True,
            **processor_args,
        )

        if self.use_cuda:
            input_values = input_values.to(self.device_map)

        # TODO: make generate generic
        logits = self.model.generate(**input_values, **generate_args)

        # Decode the model output
        transcription = self.processor.batch_decode(logits, skip_special_tokens=True)
        return transcription

    def process_wav2vec2(self, audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size):
        """
        Process audio input with the Wav2Vec2 model.
        """
        # TensorFloat32 tensor cores for float32 matrix multiplication availabl
        torch.set_float32_matmul_precision("high")

        # Split audio input into chunks with overlap
        if chunk_size > 0:
            chunks = [
                audio_input[:, i : i + chunk_size] for i in range(0, audio_input.shape[1], chunk_size - overlap_size)
            ]
        else:
            chunks = [audio_input]

        transcriptions = []
        for chunk in chunks:
            print(chunk.shape)

            processed = self.processor(
                chunk.squeeze(0),
                return_tensors="pt",
                sampling_rate=model_sampling_rate,
                truncation=False,
                padding="longest",
                do_normalize=True,
                **processor_args,
            )

            if self.use_cuda:
                input_values = processed.input_values.to(self.device_map)
                if hasattr(processed, "attention_mask"):
                    attention_mask = processed.attention_mask.to(self.device_map)

            if self.model.config.feat_extract_norm == "layer":
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits

            print(logits.shape)
            predicted_ids = torch.argmax(logits[0], dim=-1)

            # Decode each chunk
            chunk_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcriptions.append(chunk_transcription[0])
            print(chunk_transcription)

        return " ".join(transcriptions)

    def decode_audio(self, audio_data: str) -> Tuple[torch.Tensor, int]:
        """
        Decodes the base64 encoded audio data to bytes, determines its sampling rate,
        and converts it to a uniform format.

        Args:
            audio_data (str): Base64 encoded audio data.

        Returns:
            torch.Tensor: Decoded and converted audio data as a tensor.
            int: The sampling rate of the audio file.
        """
        audio_bytes = base64.b64decode(audio_data)
        audio_stream = io.BytesIO(audio_bytes)

        # Load audio in its original format
        audio = AudioSegment.from_file(audio_stream)

        # Get the sampling rate of the audio file
        original_sampling_rate = audio.frame_rate
        self.log.info(f"Original Sampling Rate: {original_sampling_rate}, total length: {audio.duration_seconds}")

        # Convert to mono (if not already)
        if audio.channels > 1:
            self.log.info(f"Converting audio from {audio.channels} to mono")
            audio = audio.set_channels(1)

        # Export to a uniform format (e.g., WAV) keeping original sampling rate
        audio_stream = io.BytesIO()
        audio.export(audio_stream, format="wav")
        audio_stream.seek(0)

        # Load the audio into a tensor
        waveform, _ = torchaudio.load(audio_stream, backend="ffmpeg")

        return waveform, original_sampling_rate
