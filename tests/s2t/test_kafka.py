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

# test_s2t_kafka.py

import pytest
from geniusrise.core import InMemoryState, StreamingInput, StreamingOutput

from geniusrise_audio.s2t.kafka import SpeechToTextKafka


@pytest.fixture
def s2t_kafka():
    input_data = [
        {"audio": b"mock_audio_data_1", "metadata": {"key": "value1"}},
        {"audio": b"mock_audio_data_2", "metadata": {"key": "value2"}},
    ]
    output_data = []

    input = StreamingInput(input_data)
    output = StreamingOutput(output_data)
    state = InMemoryState()

    s2t_kafka = SpeechToTextKafka(
        input=input,
        output=output,
        state=state,
        model_name="facebook/wav2vec2-base-960h",
        model_class="Wav2Vec2ForCTC",
        processor_class="Wav2Vec2Processor",
    )
    return s2t_kafka


def test_transcribe_stream(s2t_kafka):
    s2t_kafka.model = None
    s2t_kafka.processor = None
    s2t_kafka.process_wav2vec2 = lambda *args, **kwargs: {"transcription": "Test transcription", "segments": []}

    s2t_kafka.transcribe_stream()

    assert len(s2t_kafka.output.output_data) == 2
    assert s2t_kafka.output.output_data[0]["transcription"] == "Test transcription"
    assert s2t_kafka.output.output_data[0]["metadata"] == {"key": "value1"}
    assert s2t_kafka.output.output_data[1]["transcription"] == "Test transcription"
    assert s2t_kafka.output.output_data[1]["metadata"] == {"key": "value2"}
