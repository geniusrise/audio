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

# test_s2t_bulk.py

import os
import tempfile

import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState

from geniusrise_audio.s2t.bulk import SpeechToTextBulk


@pytest.fixture
def s2t_bulk():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    s2t_bulk = SpeechToTextBulk(
        input=input,
        output=output,
        state=state,
    )
    yield s2t_bulk

    os.rmdir(input_dir)
    os.rmdir(output_dir)


def test_transcribe(s2t_bulk):
    model_name = "facebook/wav2vec2-base-960h"
    processor_name = "facebook/wav2vec2-base-960h"
    batch_size = 2

    os.makedirs(s2t_bulk.input.input_folder, exist_ok=True)
    with open(os.path.join(s2t_bulk.input.input_folder, "audio1.wav"), "wb") as f:
        f.write(b"mock_audio_data_1")
    with open(os.path.join(s2t_bulk.input.input_folder, "audio2.wav"), "wb") as f:
        f.write(b"mock_audio_data_2")

    s2t_bulk.load_models = lambda *args, **kwargs: (None, None)
    s2t_bulk.process_whisper = lambda *args, **kwargs: {"transcription": "Test transcription", "segments": []}

    s2t_bulk.transcribe(
        model_name=model_name,
        processor_name=processor_name,
        batch_size=batch_size,
    )

    output_files = os.listdir(s2t_bulk.output.output_folder)
    assert len(output_files) == 1
    assert output_files[0].startswith("predictions-")
