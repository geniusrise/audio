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

import cherrypy
import torch
from geniusrise import BatchInput, BatchOutput, State
from pydub import AudioSegment
from transformers import AutoModel, AutoProcessor

from geniusrise_audio.base import AudioAPI


class TextToSpeechAPI(AudioAPI):
    """
    TextToSpeechAPI is a subclass of AudioAPI specifically designed for text-to-speech models.
    It extends the functionality to handle text-to-speech processing using various TTS models.

    Attributes:
        model (AutoModelForConditionalGeneration): The speech-to-text model.
        processor (AutoProcessor): The processor to prepare input audio data for the model.
    """

    model: AutoModel
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
    def generate(self):
        """
        API endpoint to generate speech from the given text using the text-to-speech model.
        Expects a JSON input with 'text' as a key containing the text to be synthesized.

        Returns:
            Dict[str, Any]: A dictionary containing the generated audio data.
        """
        input_json = cherrypy.request.json
        text = input_json.get("text")
        processor_args = input_json.get("processor_args", {})
        generate_args = input_json.get("generate_args", {})

        if not text:
            raise cherrypy.HTTPError(400, "No text provided.")

        # Preprocess and generate speech
        inputs = self.processor(text, return_tensors="pt", **processor_args)
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_ids, **generate_args)

        # Convert generated ids to audio waveform
        audio = self.processor.decode(generated_ids, output_format="wav")
        sample_rate = self.model.config.sampling_rate

        # Convert the audio to wav with sampling rate
        audio_segment = AudioSegment(audio, frame_rate=sample_rate, channels=1, sample_width=2)
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"audio": audio_base64, "text": text}
