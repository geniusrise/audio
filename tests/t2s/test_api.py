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

# test_t2s_api.py

import base64

import cherrypy
import numpy as np
import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState

from geniusrise_audio.t2s.api import TextToSpeechAPI


@pytest.fixture
def t2s_api():
    input_dir = "./input_dir"
    output_dir = "./output_dir"

    input = BatchInput(input_dir, "geniusrise-test-bucket", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test-bucket", "api_output")
    state = InMemoryState(1)

    t2s_api = TextToSpeechAPI(
        input=input,
        output=output,
        state=state,
    )
    return t2s_api


def test_synthesize(t2s_api):
    text_data = "Test text"
    output_type = "mp3"
    input_json = {
        "text": text_data,
        "output_type": output_type,
    }

    cherrypy.request.json = input_json

    t2s_api.model.config.model_type = "vits"
    t2s_api.process_mms = lambda *args, **kwargs: np.array([0.1, 0.2, 0.3])

    result = t2s_api.synthesize()

    assert "audio_file" in result
    assert "input" in result
    assert result["input"] == text_data

    audio_base64 = result["audio_file"]
    audio_data = base64.b64decode(audio_base64)
    assert len(audio_data) > 0


def test_tts_pipeline(t2s_api):
    text_data = "Test text"
    input_json = {
        "text": text_data,
    }

    cherrypy.request.json = input_json

    t2s_api.initialize_pipeline = lambda *args, **kwargs: None
    t2s_api.hf_pipeline = lambda *args, **kwargs: np.array([0.1, 0.2, 0.3])

    result = t2s_api.tts_pipeline()

    assert "audio_file" in result
    assert "input" in result
    assert result["input"] == text_data
