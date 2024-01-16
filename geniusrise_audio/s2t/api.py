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

import cherrypy
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from geniusrise import BatchInput, BatchOutput, State
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
        model_name: str,
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
        input_sampling_rate = input_json.get("input_sampling_rate")
        model_sampling_rate = input_json.get("model_sampling_rate")
        processor_args = input_json.get("processor_args")
        generate_args = input_json.get("generate_args")

        if not audio_data:
            raise cherrypy.HTTPError(400, "No audio data provided.")

        # Convert base64 encoded data to bytes
        audio_input = self.decode_audio(audio_data)
        audio_input = torchaudio.functional.resample(
            audio_input, orig_freq=input_sampling_rate, new_freq=model_sampling_rate
        )

        # Preprocess and transcribe
        input_values = self.processor(
            audio_input, return_tensors="pt", sampling_rate=model_sampling_rate, **processor_args
        ).input_values

        # Perform inference
        with torch.no_grad():
            logits = self.model.generate(input_values, **generate_args).logits

        # Decode the model output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])

        return {"transcription": transcription}

    def decode_audio(self, audio_data: str) -> bytes:
        """
        Decodes the base64 encoded audio data to bytes.

        Args:
            audio_data (str): Base64 encoded audio data.

        Returns:
            bytes: Decoded audio data.
        """
        import base64

        return base64.b64decode(audio_data)
