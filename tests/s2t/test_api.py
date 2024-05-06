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

# test_s2t_api.py

import base64

import cherrypy
import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState

from geniusrise_audio.s2t.api import SpeechToTextAPI


@pytest.fixture
def s2t_api():
    input_dir = "./input_dir"
    output_dir = "./output_dir"

    input = BatchInput(input_dir, "geniusrise-test-bucket", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "api_output")
    state = InMemoryState(1)

    s2t_api = SpeechToTextAPI(
        input=input,
        output=output,
        state=state,
    )
    return s2t_api


def test_transcribe(s2t_api):
    audio_data = b"mock_audio_data"
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    input_json = {
        "audio_file": audio_base64,
        "model_sampling_rate": 16000,
    }

    cherrypy.request.json = input_json

    s2t_api.load_models = lambda *args, **kwargs: (None, None)
    s2t_api.model.config.model_type = "wav2vec2"
    s2t_api.process_wav2vec2 = lambda *args, **kwargs: {"transcription": "Test transcription", "segments": []}

    result = s2t_api.transcribe()

    assert "transcriptions" in result
    assert result["transcriptions"] == "Test transcription"
