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

# test_s2t_inference.py

import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState, StreamingInput, StreamingOutput

from geniusrise_audio.s2t.inference import SpeechToTextInference, SpeechToTextInferenceStream


@pytest.fixture
def s2t_inference():
    input_dir = "./input_dir"
    output_dir = "./output_dir"

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    s2t_inference = SpeechToTextInference(
        input=input,
        output=output,
        state=state,
    )
    return s2t_inference


@pytest.fixture
def s2t_inference_stream():
    input_data = [
        {"audio": b"mock_audio_data_1", "metadata": {"key": "value1"}},
        {"audio": b"mock_audio_data_2", "metadata": {"key": "value2"}},
    ]
    output_data = []

    input = StreamingInput(input_data)
    output = StreamingOutput(output_data)
    state = InMemoryState()

    s2t_inference_stream = SpeechToTextInferenceStream(
        input=input,
        output=output,
        state=state,
    )
    return s2t_inference_stream


def test_process_faster_whisper(s2t_inference):
    audio_input = b"mock_audio_data"
    model_sampling_rate = 16000
    chunk_size = 0
    generate_args = {}

    s2t_inference.model = None
    s2t_inference.model.transcribe = lambda *args, **kwargs: (["Test transcription"], {"key": "value"})

    result = s2t_inference.process_faster_whisper(audio_input, model_sampling_rate, chunk_size, generate_args)

    assert result["transcriptions"] == ["Test transcription"]
    assert result["transcription_info"] == {"key": "value"}


def test_process_wav2vec2(s2t_inference_stream):
    audio_input = b"mock_audio_data"
    model_sampling_rate = 16000
    processor_args = {}
    chunk_size = 0
    overlap_size = 0

    s2t_inference_stream.model = None
    s2t_inference_stream.model.config.feat_extract_norm = "layer"
    s2t_inference_stream.model.return_value = None
    s2t_inference_stream.model.return_value.logits = "Test logits"
    s2t_inference_stream.processor.batch_decode = lambda *args, **kwargs: ["Test transcription"]

    result = s2t_inference_stream.process_wav2vec2(
        audio_input, model_sampling_rate, processor_args, chunk_size, overlap_size
    )

    assert result["transcription"] == "Test transcription"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["tokens"] == "Test transcription"
