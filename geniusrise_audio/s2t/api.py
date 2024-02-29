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
import multiprocessing
import numpy as np
import torch
from typing import Any, Dict, Union
from geniusrise import BatchInput, BatchOutput, State
from transformers import AutoModelForCTC, AutoProcessor, pipeline

from geniusrise_audio.base import AudioAPI
from geniusrise_audio.s2t.util import whisper_alignment_heads, decode_audio, chunk_audio


class SpeechToTextAPI(AudioAPI):
    r"""
    SpeechToTextAPI is a subclass of AudioAPI specifically designed for speech-to-text models.
    It extends the functionality to handle speech-to-text processing using various ASR models.

    Attributes:
        model (AutoModelForCTC): The speech-to-text model.
        processor (AutoProcessor): The processor to prepare input audio data for the model.

    Methods:
        transcribe(audio_input: bytes) -> str:
            Transcribes the given audio input to text using the speech-to-text model.

    Example CLI Usage:

    ```bash
    genius SpeechToTextAPI rise \
    batch \
        --input_folder ./input \
    batch \
        --output_folder ./output \
    none \
        --id facebook/wav2vec2-large-960h-lv60-self \
        listen \
            --args \
                model_name="facebook/wav2vec2-large-960h-lv60-self" \
                model_class="Wav2Vec2ForCTC" \
                processor_class="Wav2Vec2Processor" \
                use_cuda=True \
                precision="float32" \
                quantization=0 \
                device_map="cuda:0" \
                max_memory=None \
                torchscript=False \
                compile=True \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```

    or using whisper.cpp:

    ```bash
    genius SpeechToTextAPI rise \
        batch \
                --input_folder ./input \
        batch \
                --output_folder ./output \
        none \
        listen \
            --args \
                model_name="large" \
                use_whisper_cpp=True \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```
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
        r"""
        API endpoint to transcribe the given audio input to text using the speech-to-text model.
        Expects a JSON input with 'audio_file' as a key containing the base64 encoded audio data.

        Returns:
            Dict[str, str]: A dictionary containing the transcribed text.

        Example CURL Request for transcription:
        ```bash
        (base64 -w 0 sample.flac | awk '{print "{\"audio_file\": \""$0"\", \"model_sampling_rate\": 16000, \"chunk_size\": 1280000, \"overlap_size\": 213333, \"do_sample\": true, \"num_beams\": 4, \"temperature\": 0.6, \"tgt_lang\": \"eng\"}"}' > /tmp/payload.json)
        curl -X POST http://localhost:3000/api/v1/transcribe \
            -H "Content-Type: application/json" \
            -u user:password \
            -d @/tmp/payload.json | jq
        ```
        """
        input_json = cherrypy.request.json
        audio_data = input_json.get("audio_file")
        model_sampling_rate = input_json.get("model_sampling_rate", 16_000)
        processor_args = input_json.get("processor_args", {})
        chunk_size = input_json.get("chunk_size", 0)
        overlap_size = input_json.get("overlap_size", 0)

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

        if chunk_size > 0 and overlap_size == 0:
            overlap_size = int(chunk_size / 6)

        # TODO: support voice presets

        if not audio_data:
            raise cherrypy.HTTPError(400, "No audio data provided.")

        # Convert base64 encoded data to bytes
        audio_bytes = base64.b64decode(audio_data)
        audio_input, input_sampling_rate = decode_audio(
            audio_bytes=audio_bytes,
            model_type=self.model.config.model_type,
            model_sampling_rate=model_sampling_rate,
        )

        # Perform inference
        with torch.no_grad():
            if self.use_whisper_cpp:
                transcription = self.model.transcribe(audio_input, num_proc=multiprocessing.cpu_count())
            elif self.use_faster_whisper:
                transcription = self.process_faster_whisper(audio_input, model_sampling_rate, chunk_size, generate_args)
            elif self.model.config.model_type == "whisper":
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

        return {"transcriptions": transcription}

    def process_faster_whisper(
        self,
        audio_input: Union[str, bytes, np.ndarray],
        model_sampling_rate: int,
        chunk_size: int,
        generate_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Processes audio input with the faster-whisper model.

        Args:
            audio_input (Union[str, bytes, np.ndarray]): The audio input for transcription.
            model_sampling_rate (int): The sampling rate of the model.
            chunk_size (int): The size of audio chunks to process.
            generate_args (Dict[str, Any]): Additional arguments for transcription.

        Returns:
            Dict[str, Any]: A dictionary containing the transcription results.
        """
        transcribed_segments, transcription_info = self.model.transcribe(
            beam_size=generate_args.get("beam_size", 5),
            best_of=generate_args.get("best_of", 5),
            patience=generate_args.get("patience", 1.0),
            length_penalty=generate_args.get("length_penalty", 1.0),
            repetition_penalty=generate_args.get("repetition_penalty", 1.0),
            no_repeat_ngram_size=generate_args.get("no_repeat_ngram_size", 0),
            temperature=generate_args.get("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            compression_ratio_threshold=generate_args.get("compression_ratio_threshold", 2.4),
            log_prob_threshold=generate_args.get("log_prob_threshold", -1.0),
            no_speech_threshold=generate_args.get("no_speech_threshold", 0.6),
            condition_on_previous_text=generate_args.get("condition_on_previous_text", True),
            prompt_reset_on_temperature=generate_args.get("prompt_reset_on_temperature", 0.5),
            initial_prompt=generate_args.get("initial_prompt", None),
            prefix=generate_args.get("prefix", None),
            suppress_blank=generate_args.get("suppress_blank", True),
            suppress_tokens=generate_args.get("suppress_tokens", [-1]),
            without_timestamps=generate_args.get("without_timestamps", False),
            max_initial_timestamp=generate_args.get("max_initial_timestamp", 1.0),
            word_timestamps=generate_args.get("word_timestamps", False),
            prepend_punctuations=generate_args.get("prepend_punctuations", "\"'“¿([{-"),
            append_punctuations=generate_args.get(
                "append_punctuations",
                "\"'.。,，!！?？:：”)]}、",
            ),
            vad_filter=generate_args.get("vad_filter", False),
            vad_parameters=generate_args.get("vad_parameters", None),
            max_new_tokens=generate_args.get("max_new_tokens", None),
            chunk_length=generate_args.get("chunk_length", None),
            clip_timestamps=generate_args.get("clip_timestamps", "0"),
            hallucination_silence_threshold=generate_args.get("hallucination_silence_threshold", None),
        )

        # Format the results
        transcriptions = [segment.text for segment in transcribed_segments]
        return {"transcriptions": transcriptions, "transcription_info": transcription_info._asdict()}

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
            **input_values,
            **generate_args,  # , return_timestamps=True, return_token_timestamps=True, return_segments=True
        )

        # Decode the model output
        if type(logits) is torch.Tensor:
            transcription = self.processor.batch_decode(logits[0], skip_special_tokens=True)
            return {"transcription": "".join(transcription), "segments": []}
        else:
            transcription = self.processor.batch_decode(logits["sequences"], skip_special_tokens=True)
            segments = self.processor.batch_decode(
                [x["tokens"] for x in logits["segments"][0]], skip_special_tokens=True
            )
            timestamps = [
                {
                    "tokens": t,
                    "start": l["start"].cpu().numpy().tolist(),
                    "end": l["end"].cpu().numpy().tolist(),
                }
                for t, l in zip(segments, logits["segments"][0])
            ]
            return {"transcription": transcription, "segments": timestamps}

    def process_seamless(
        self, audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size, generate_args
    ):
        """
        Process audio input with the Whisper model.
        """
        audio_input = audio_input.squeeze(0)

        # Split audio input into chunks with overlap
        chunks = chunk_audio(audio_input, chunk_size, overlap_size, overlap_size) if chunk_size > 0 else [audio_input]

        segments = []
        for chunk_id, chunk in enumerate(chunks):
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
            segments.append(
                {
                    "tokens": " ".join([x.strip() for x in _transcription]).strip(),
                    "start": chunk_id * overlap_size,
                    "end": (chunk_id + 1) * overlap_size,
                }
            )

        transcription = " ".join([s["tokens"].strip() for s in segments])
        return {"transcription": transcription, "segments": segments}

    def process_wav2vec2(self, audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size):
        """
        Process audio input with the Wav2Vec2 model.
        """
        # TensorFloat32 tensor cores for float32 matrix multiplication availabl
        torch.set_float32_matmul_precision("high")
        audio_input = audio_input.squeeze(0)

        # Split audio input into chunks with overlap
        chunks = chunk_audio(audio_input, chunk_size, overlap_size, overlap_size) if chunk_size > 0 else [audio_input]

        segments = []
        for chunk_id, chunk in enumerate(chunks):
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
            else:
                input_values = processed.input_values
                if hasattr(processed, "attention_mask"):
                    attention_mask = processed.attention_mask

            if self.model.config.feat_extract_norm == "layer":
                logits = self.model(input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode each chunk
            chunk_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            segments.append(
                {
                    "tokens": chunk_transcription[0],
                    "start": chunk_id * overlap_size,
                    "end": (chunk_id + 1) * overlap_size,
                }
            )

        transcription = " ".join([s["tokens"].strip() for s in segments])
        return {"transcription": transcription, "segments": segments}

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
        r"""
        Recognizes named entities in the input text using the Hugging Face pipeline.

        This method leverages a pre-trained NER model to identify and classify entities in text into categories such as
        names, organizations, locations, etc. It's suitable for processing various types of text content.

        Args:
            **kwargs (Any): Arbitrary keyword arguments, typically containing 'text' for the input text.

        Returns:
            Dict[str, Any]: A dictionary containing the original input text and a list of recognized entities.

        Example CURL Request for transcription:
        ```bash
        (base64 -w 0 sample.flac | awk '{print "{\"audio_file\": \""$0"\", \"model_sampling_rate\": 16000, \"chunk_length_s\": 60}"}' > /tmp/payload.json)
        curl -X POST http://localhost:3000/api/v1/asr_pipeline \
            -H "Content-Type: application/json" \
            -u user:password \
            -d @/tmp/payload.json | jq
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

        audio_input, input_sampling_rate = decode_audio(audio_data, self.model.config.model_type, model_sampling_rate)

        result = self.hf_pipeline(audio_input.squeeze(0).numpy(), **generate_args)  # type: ignore

        return {"transcriptions": result}
