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

# test_t2s_inference.py

import numpy as np
import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState, StreamingInput, StreamingOutput

from geniusrise_audio.t2s.inference import TextToSpeechInference, TextToSpeechInferenceStream


@pytest.fixture
def t2s_inference():
    input_dir = "./input_dir"
    output_dir = "./output_dir"

    input = BatchInput(input_dir, "geniusrise-test-bucket", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "api_output")
    state = InMemoryState(1)

    t2s_inference = TextToSpeechInference(
        input=input,
        output=output,
        state=state,
    )
    return t2s_inference


@pytest.fixture
def t2s_inference_stream():
    input_data = [
        {"text": "Test text 1", "voice_preset": "0", "metadata": {"key": "value1"}},
        {"text": "Test text 2", "voice_preset": "1", "metadata": {"key": "value2"}},
    ]
    output_data = []

    input = StreamingInput(input_data)
    output = StreamingOutput(output_data)
    state = InMemoryState(1)

    t2s_inference_stream = TextToSpeechInferenceStream(
        input=input,
        output=output,
        state=state,
    )
    return t2s_inference_stream


def test_process_mms(t2s_inference):
    text_input = "Test text"
    generate_args = {}

    t2s_inference.processor = lambda *args, **kwargs: "Test inputs"
    t2s_inference.model = None
    t2s_inference.model.generate = lambda *args, **kwargs: None
    t2s_inference.model.return_value = None
    t2s_inference.model.return_value.waveform = [np.array([0.1, 0.2, 0.3])]

    result = t2s_inference.process_mms(text_input, generate_args)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_process_bark(t2s_inference_stream):
    text_input = "Test text"
    voice_preset = "0"
    generate_args = {}

    t2s_inference_stream.processor = lambda *args, **kwargs: "Test inputs"
    t2s_inference_stream.model = None
    t2s_inference_stream.model.generate = lambda *args, **kwargs: np.array([0.1, 0.2, 0.3])

    result = t2s_inference_stream.process_bark(text_input, voice_preset, generate_args)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
