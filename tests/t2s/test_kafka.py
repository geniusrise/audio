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

# test_t2s_kafka.py

import base64

import numpy as np
import pytest
from geniusrise.core import InMemoryState, StreamingInput, StreamingOutput

from geniusrise_audio.t2s.kafka import TextToSpeechKafka


@pytest.fixture
def t2s_kafka():
    input_data = [
        {"text": "Test text 1", "voice_preset": "0", "metadata": {"key": "value1"}},
        {"text": "Test text 2", "voice_preset": "1", "metadata": {"key": "value2"}},
    ]
    output_data = []

    input = StreamingInput(input_data)
    output = StreamingOutput(output_data)
    state = InMemoryState(1)

    t2s_kafka = TextToSpeechKafka(
        input=input,
        output=output,
        state=state,
        model_name="facebook/fastspeech2-en-ljspeech",
        model_class="SpeechSynthesisModel",
        processor_class="SpeechSynthesisProcessor",
    )
    return t2s_kafka


def test_synthesize_stream(t2s_kafka):
    t2s_kafka.model = None
    t2s_kafka.processor = None
    t2s_kafka.process_mms = lambda *args, **kwargs: np.array([0.1, 0.2, 0.3])

    t2s_kafka.synthesize_stream()

    assert len(t2s_kafka.output.output_data) == 2
    assert "audio" in t2s_kafka.output.output_data[0]
    assert "metadata" in t2s_kafka.output.output_data[0]
    assert t2s_kafka.output.output_data[0]["metadata"] == {"key": "value1"}
    assert "audio" in t2s_kafka.output.output_data[1]
    assert "metadata" in t2s_kafka.output.output_data[1]
    assert t2s_kafka.output.output_data[1]["metadata"] == {"key": "value2"}

    audio_base64_1 = t2s_kafka.output.output_data[0]["audio"]
    audio_data_1 = base64.b64decode(audio_base64_1)
    assert len(audio_data_1) > 0

    audio_base64_2 = t2s_kafka.output.output_data[1]["audio"]
    audio_data_2 = base64.b64decode(audio_base64_2)
    assert len(audio_data_2) > 0
