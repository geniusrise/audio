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
import cherrypy
import torch
import numpy as np
from typing import Any, Dict, List
from geniusrise import BatchInput, BatchOutput, State
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, SpeechT5HifiGan
from datasets import load_dataset

from geniusrise_audio.base import AudioAPI
from geniusrise_audio.t2s.util import convert_waveform_to_audio_file


class TextToSpeechAPI(AudioAPI):
    """
    TextToSpeechAPI for converting text to speech using various TTS models.

    Attributes:
        model (AutoModelForSeq2SeqLM): The text-to-speech model.
        tokenizer (AutoTokenizer): The tokenizer for the model.

    Methods:
        synthesize(text_input: str) -> bytes:
            Converts the given text input to speech using the text-to-speech model.

    Example CLI Usage:
    ... [Similar to SpeechToTextAPI example CLI usage] ...
    """

    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the TextToSpeechAPI with configurations for text-to-speech processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)
        self.hf_pipeline = None
        self.vocoder = None
        self.embeddings_dataset = None

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def synthesize(self):
        """
        API endpoint to convert text input to speech using the text-to-speech model.
        Expects a JSON input with 'text' as a key containing the text to be synthesized.

        Returns:
            Dict[str, str]: A dictionary containing the base64 encoded audio data.

        Example CURL Request for synthesis:
        ... [Provide example CURL request] ...
        """
        input_json = cherrypy.request.json
        text_data = input_json.get("text")
        output_type = input_json.get("output_type")
        voice_preset = input_json.get("voice_preset")

        generate_args = input_json.copy()

        if "text" in generate_args:
            del generate_args["text"]
        if "output_type" in generate_args:
            del generate_args["output_type"]
        if "voice_preset" in generate_args:
            del generate_args["voice_preset"]

        if not text_data:
            raise cherrypy.HTTPError(400, "No text data provided.")

        # Perform inference
        if self.model.config.model_type == "vits":
            audio_output = self.process_mms(text_data, generate_args=generate_args)
        elif self.model.config.model_type == "coarse_acoustics" or self.model.config.model_type == "bark":
            audio_output = self.process_bark(text_data, voice_preset=voice_preset, generate_args=generate_args)
        elif self.model.config.model_type == "speecht5":
            audio_output = self.process_speecht5_tts(text_data, voice_preset=voice_preset, generate_args=generate_args)
        elif self.model.config.model_type == "seamless_m4t_v2":
            audio_output = self.process_seamless(text_data, voice_preset=voice_preset, generate_args=generate_args)

        # Convert audio to base64 encoded data
        sample_rate = (
            self.model.generation_config.sample_rate if hasattr(self.model.generation_config, "sample_rate") else 16_000
        )
        audio_file = convert_waveform_to_audio_file(audio_output, format=output_type, sample_rate=sample_rate)
        audio_base64 = base64.b64encode(audio_file)

        return {"audio_file": audio_base64.decode("utf-8"), "input": text_data}

    def process_mms(self, text_input: str, generate_args: dict) -> np.ndarray:
        inputs = self.processor(text_input, return_tensors="pt")

        if self.use_cuda:
            inputs = inputs.to(self.device_map)

        with torch.no_grad():
            outputs = self.model(**inputs, **generate_args)

        waveform = outputs.waveform[0].cpu().numpy().squeeze()
        return waveform

    def process_bark(self, text_input: str, voice_preset: str, generate_args: dict) -> np.ndarray:
        # Process the input text with the selected voice preset
        # Presets here: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
        chunks = text_input.split(".")
        audio_arrays: List[np.ndarray] = []
        for chunk in chunks:
            inputs = self.processor(chunk, voice_preset=voice_preset, return_tensors="pt", return_attention_mask=True)

            if self.use_cuda:
                inputs = inputs.to(self.device_map)

            # Generate the audio waveform
            with torch.no_grad():
                audio_array = self.model.generate(**inputs, **generate_args, min_eos_p=0.05)
                audio_array = audio_array.cpu().numpy().squeeze()
                audio_arrays.append(audio_array)

        return np.concatenate(audio_arrays)

    def process_speecht5_tts(self, text_input: str, voice_preset: str, generate_args: dict) -> np.ndarray:
        if not self.vocoder:
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            if self.use_cuda:
                self.vocoder = self.vocoder.to(self.device_map)  # type: ignore
        if not self.embeddings_dataset:
            # use the CMU arctic dataset for voice presets
            self.embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors", split="validation", revision="01090996e2ec93b238f194db1ff9c184ed741b07"
            )

        chunks = text_input.split(".")
        audio_arrays: List[np.ndarray] = []
        for chunk in chunks:
            inputs = self.processor(text=chunk, return_tensors="pt")
            speaker_embeddings = torch.tensor(self.embeddings_dataset[int(voice_preset)]["xvector"]).unsqueeze(0)  # type: ignore

            if self.use_cuda:
                inputs = inputs.to(self.device_map)
                speaker_embeddings = speaker_embeddings.to(self.device_map)  # type: ignore

            with torch.no_grad():
                # Generate speech tensor
                speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
                audio_output = speech.cpu().numpy().squeeze()
                audio_arrays.append(audio_output)

        return np.concatenate(audio_arrays)

    def process_seamless(self, text_input: str, voice_preset: str, generate_args: dict) -> np.ndarray:
        # Splitting the input text into chunks based on full stops to manage long text inputs
        chunks = text_input.split(".")
        audio_arrays: List[np.ndarray] = []

        for chunk in chunks:
            inputs = self.processor(text=chunk, return_tensors="pt", src_lang="eng")

            if self.use_cuda:
                inputs = inputs.to(self.device_map)

            # Generate the audio waveform
            with torch.no_grad():
                # Seamless M4T v2 specific generation code
                outputs = self.model.generate(inputs.input_ids, speaker_id=int(voice_preset), **generate_args)[0]

            audio_array = outputs.cpu().numpy().squeeze()
            audio_arrays.append(audio_array)

        return np.concatenate(audio_arrays)

    def initialize_pipeline(self):
        """
        Lazy initialization of the TTS Hugging Face pipeline.
        """
        if not self.hf_pipeline:
            self.hf_pipeline = pipeline(
                "text-to-speech",
                model=self.model,
                tokenizer=self.processor,
                framework="pt",
            )

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def tts_pipeline(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Converts text to speech using the Hugging Face pipeline.

        Args:
            **kwargs (Any): Arbitrary keyword arguments, typically containing 'text' for the input text.

        Returns:
            Dict[str, Any]: A dictionary containing the base64 encoded audio data.

        Example CURL Request for synthesis:
        ... [Provide example CURL request] ...
        """
        self.initialize_pipeline()  # Initialize the pipeline on first API hit

        input_json = cherrypy.request.json
        text_data = input_json.get("text")

        result = self.hf_pipeline(text_data, **kwargs)  # type: ignore

        # Convert audio to base64 encoded data
        audio_base64 = base64.encode(result)  # type: ignore

        return {"audio_file": audio_base64, "input": text_data}
