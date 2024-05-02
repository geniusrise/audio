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

# test_stream.py

import pytest
from geniusrise.core import InMemoryState, StreamingInput, StreamingOutput

from geniusrise_audio.base.stream import AudioStream


@pytest.fixture
def audio_stream():
    input_data = [
        {"audio": b"mock_audio_data_1", "metadata": {"key": "value1"}},
        {"audio": b"mock_audio_data_2", "metadata": {"key": "value2"}},
    ]
    output_data = []

    input = StreamingInput(input_data)
    output = StreamingOutput(output_data)
    state = InMemoryState()

    audio_stream = AudioStream(
        input=input,
        output=output,
        state=state,
    )
    return audio_stream


def test_load_models(audio_stream):
    model_name = "facebook/wav2vec2-base-960h"
    processor_name = "facebook/wav2vec2-base-960h"
    model_class = "Wav2Vec2ForCTC"
    processor_class = "Wav2Vec2Processor"

    model, processor = audio_stream.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
    )

    assert model is not None
    assert processor is not None
    assert len(list(model.named_modules())) > 0


def test_process_audio(audio_stream):
    audio_input = b"mock_audio_data"
    model_sampling_rate = 16000

    audio_stream.model = None
    audio_stream.processor = None
    audio_stream.model_sampling_rate = model_sampling_rate
    audio_stream.use_cuda = False
    audio_stream.process_wav2vec2 = lambda *args, **kwargs: {"transcription": "Test transcription", "segments": []}

    result = audio_stream.process_audio(audio_input, model_sampling_rate)

    assert result["transcription"] == "Test transcription"
    assert result["segments"] == []
