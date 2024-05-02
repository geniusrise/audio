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

# test_t2s_bulk.py

import os
import tempfile

import numpy as np
import pytest
from datasets import Dataset
from geniusrise.core import BatchInput, BatchOutput, InMemoryState

from geniusrise_audio.t2s.bulk import TextToSpeechBulk


@pytest.fixture
def t2s_bulk():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    t2s_bulk = TextToSpeechBulk(
        input=input,
        output=output,
        state=state,
    )
    yield t2s_bulk

    os.rmdir(input_dir)
    os.rmdir(output_dir)


def test_load_dataset(t2s_bulk):
    os.makedirs(t2s_bulk.input.input_folder, exist_ok=True)
    with open(os.path.join(t2s_bulk.input.input_folder, "dataset.jsonl"), "w") as f:
        f.write('{"text": "Sample text 1"}\n')
        f.write('{"text": "Sample text 2"}\n')

    dataset = t2s_bulk.load_dataset(dataset_path=t2s_bulk.input.input_folder)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0]["text"] == "Sample text 1"
    assert dataset[1]["text"] == "Sample text 2"


def test_synthesize_speech(t2s_bulk):
    os.makedirs(t2s_bulk.input.input_folder, exist_ok=True)
    with open(os.path.join(t2s_bulk.input.input_folder, "dataset.jsonl"), "w") as f:
        f.write('{"text": "Sample text 1"}\n')
        f.write('{"text": "Sample text 2"}\n')

    model_name = "facebook/fastspeech2-en-ljspeech"
    batch_size = 2

    t2s_bulk.load_models = lambda *args, **kwargs: (None, None)
    t2s_bulk.process_mms = lambda text_input, generate_args: np.array([0.1, 0.2, 0.3])

    t2s_bulk.synthesize_speech(
        model_name=model_name,
        batch_size=batch_size,
    )

    output_files = os.listdir(t2s_bulk.output.output_folder)
    assert len(output_files) == 2
    assert output_files[0].endswith(".wav") or output_files[0].endswith(".mp3")
    assert output_files[1].endswith(".wav") or output_files[1].endswith(".mp3")
