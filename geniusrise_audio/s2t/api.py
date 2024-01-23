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
from typing import Tuple, Any, Dict
import soundfile as sf
import librosa
import numpy as np
from geniusrise import BatchInput, BatchOutput, State
from transformers import AutoModelForCTC, AutoProcessor, pipeline

from geniusrise_audio.base import AudioAPI
from geniusrise_audio.s2t.util import whisper_alignment_heads


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
        self.hf_pipeline = None

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
        model_sampling_rate = input_json.get("model_sampling_rate", 16_000)
        processor_args = input_json.get("processor_args", {})
        chunk_size = input_json.get("chunk_size", 0)
        overlap_size = input_json.get("overlap_size", 1600)

        generate_args = input_json.copy()

        if "audio_file" in generate_args:
            del generate_args["audio_file"]
        if "model_sampling_rate" in generate_args:
            del generate_args["model_sampling_rate"]
        if "processor_args" in generate_args:
            del generate_args["processor_args"]
        if "chunk_size" in generate_args:
            del generate_args["chunk_size"]
        if "overlap_size" in generate_args:
            del generate_args["overlap_size"]

        # TODO: support voice presets

        if not audio_data:
            raise cherrypy.HTTPError(400, "No audio data provided.")

        # Convert base64 encoded data to bytes
        audio_input, input_sampling_rate = self.decode_audio(
            audio_data, self.model.config.model_type, model_sampling_rate
        )

        # Perform inference
        with torch.no_grad():
            if self.model.config.model_type == "whisper":
                transcription = self.process_whisper(
                    audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size, generate_args
                )
            elif self.model.config.model_type == "seamless_m4t_v2":
                transcription = self.process_seamless(
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
        alignment_heads = [v for k, v in whisper_alignment_heads.items() if k in self.model_name][0]
        self.model.generation_config.alignment_heads = alignment_heads

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
        logits = self.model.generate(
            **input_values, **generate_args, return_timestamps=True, return_token_timestamps=True, return_segments=True
        )

        # Decode the model output
        transcription = self.processor.batch_decode(logits["sequences"], skip_special_tokens=True)
        segments = self.processor.batch_decode([x["tokens"] for x in logits["segments"][0]], skip_special_tokens=True)
        timestamps = [
            {
                "tokens": t,
                "start": l["start"].cpu().numpy().tolist(),
                "end": l["end"].cpu().numpy().tolist(),
            }
            for t, l in zip(segments, logits["segments"][0])
        ]
        return {"text": transcription, "segments": timestamps}

    def process_seamless(
        self, audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size, generate_args
    ):
        """
        Process audio input with the Whisper model.
        """
        audio_input = audio_input.squeeze(0)

        # Split audio input into chunks with overlap
        chunks = (
            self.chunk_audio(audio_input.squeeze(0), chunk_size, overlap_size, overlap_size)
            if chunk_size > 0
            else [audio_input]
        )

        transcriptions = []
        for chunk in chunks:
            # Preprocess and transcribe
            input_values = self.processor(
                audios=chunk,
                return_tensors="pt",
                sampling_rate=model_sampling_rate,
                do_normalize=True,
                **processor_args,
            )

            if self.use_cuda:
                input_values = input_values.to(self.device_map)

            # TODO: make generate generic
            logits = self.model.generate(**input_values, **generate_args)[0]

            # Decode the model output
            _transcription = self.processor.batch_decode(logits, skip_special_tokens=True)
            transcriptions.append(" ".join(_transcription))
        return " ".join(transcriptions)

    def process_wav2vec2(self, audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size):
        """
        Process audio input with the Wav2Vec2 model.
        """
        # TensorFloat32 tensor cores for float32 matrix multiplication availabl
        torch.set_float32_matmul_precision("high")
        audio_input = audio_input.squeeze(0)

        # Split audio input into chunks with overlap
        chunks = (
            self.chunk_audio(audio_input.squeeze(0), chunk_size, overlap_size, overlap_size)
            if chunk_size > 0
            else [audio_input]
        )

        transcriptions = []
        for chunk in chunks:
            print(chunk.shape)

            processed = self.processor(
                chunk,
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
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode each chunk
            chunk_transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
            transcriptions.append(chunk_transcription)
            print(chunk_transcription.strip())

        return " ".join(transcriptions)

    def chunk_audio(self, audio_input, chunk_size, stride_left, stride_right):
        """
        Splits the audio input into overlapping chunks with specified left and right strides.

        Args:
            audio_input (torch.Tensor): The input audio tensor.
            chunk_size (int): The size of each audio chunk.
            stride_left (int): The size of the left stride for overlap.
            stride_right (int): The size of the right stride for overlap.

        Returns:
            List[torch.Tensor]: List of chunked audio tensors with overlap.
        """
        chunks = []

        for block_start in range(0, len(audio_input), chunk_size):
            chunk_end_idx = min(block_start + chunk_size + stride_right, len(audio_input))
            chunk_start_idx = max(0, block_start - stride_left)

            chunk = audio_input[chunk_start_idx:chunk_end_idx]
            chunks.append(chunk)

        return chunks

    def decode_audio(self, audio_data: str, model_type: str, model_sampling_rate: int) -> Tuple[torch.Tensor, int]:
        """
        Decodes the base64 encoded audio data to bytes, determines its sampling rate,
        and converts it to a uniform format based on the model type.

        Args:
            audio_data (str): Base64 encoded audio data.
            model_type (str): The type of model to be used for transcription.

        Returns:
            torch.Tensor: Decoded and converted audio data as a tensor.
            int: The sampling rate of the audio file.
        """
        audio_bytes = base64.b64decode(audio_data)
        audio_stream = io.BytesIO(audio_bytes)

        if model_type == "wav2vec":
            # Using Librosa for Wave2Vec
            _waveform, original_sampling_rate = librosa.load(audio_stream, sr=model_sampling_rate, mono=True)
            waveform = torch.from_numpy(_waveform).unsqueeze(0)

        elif model_type == "wav2vec2":
            # Using soundfile for Wave2Vec2
            _waveform, original_sampling_rate = sf.read(audio_stream, dtype="float32")

            # Convert to mono if stereo
            if _waveform.ndim == 2:
                _waveform = np.mean(_waveform, axis=1)

            # Resample to 16kHz if needed
            if original_sampling_rate != model_sampling_rate:
                _waveform = librosa.resample(_waveform, orig_sr=original_sampling_rate, target_sr=model_sampling_rate)

            waveform = torch.from_numpy(_waveform).unsqueeze(0)

        else:
            # For whisper, seamlessm4t
            # Using PyDub for other models
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

            waveform = torchaudio.functional.resample(
                waveform, orig_freq=original_sampling_rate, new_freq=model_sampling_rate
            )

        return waveform, int(original_sampling_rate)

    def initialize_pipeline(self):
        """
        Lazy initialization of the NER Hugging Face pipeline.
        """
        if not self.hf_pipeline:
            self.hf_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                feature_extractor=self.processor.feature_extractor,
                tokenizer=self.processor.tokenizer,
                framework="pt",
            )

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def asr_pipeline(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Recognizes named entities in the input text using the Hugging Face pipeline.

        This method leverages a pre-trained NER model to identify and classify entities in text into categories such as
        names, organizations, locations, etc. It's suitable for processing various types of text content.

        Args:
            **kwargs (Any): Arbitrary keyword arguments, typically containing 'text' for the input text.

        Returns:
            Dict[str, Any]: A dictionary containing the original input text and a list of recognized entities.

        Example CURL Request for NER:
        ```bash
        curl -X POST localhost:3000/api/v1/ner_pipeline \
            -H "Content-Type: application/json" \
            -d '{"text": "John Doe works at OpenAI in San Francisco."}' | jq
        ```
        """
        self.initialize_pipeline()  # Initialize the pipeline on first API hit

        input_json = cherrypy.request.json
        audio_data = input_json.get("audio_file")
        model_sampling_rate = input_json.get("model_sampling_rate", 16_000)

        generate_args = input_json.copy()

        if "audio_file" in generate_args:
            del generate_args["audio_file"]
        if "model_sampling_rate" in generate_args:
            del generate_args["model_sampling_rate"]

        audio_input, input_sampling_rate = self.decode_audio(
            audio_data, self.model.config.model_type, model_sampling_rate
        )

        result = self.hf_pipeline(audio_input.squeeze(0).numpy(), **generate_args)  # type: ignore

        return {"transcriptions": result}
