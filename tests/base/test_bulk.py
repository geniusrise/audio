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

# test_base_bulk.py

import os

import pytest
import torch
from geniusrise.core import BatchInput, BatchOutput, InMemoryState

from geniusrise_audio.base.bulk import AudioBulk


@pytest.fixture(
    # fmt: off
    params=[
        ("facebook/wav2vec2-base-960h", "facebook/wav2vec2-base-960h", "Wav2Vec2ForCTC", "Wav2Vec2Processor", True, "float32", 0, "cuda:0", None, False),
        ("facebook/wav2vec2-base-960h", "facebook/wav2vec2-base-960h", "Wav2Vec2ForCTC", "Wav2Vec2Processor", False, "float32", 0, None, None, False),
    ]
    # fmt: on
)
def model_config(request):
    return request.param


@pytest.fixture
def audio_bulk():
    input_dir = "./input_dir"
    output_dir = "./output_dir"

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    audio_bulk = AudioBulk(
        input=input,
        output=output,
        state=state,
    )
    yield audio_bulk

    if os.path.exists(input_dir):
        os.rmdir(input_dir)
    if os.path.exists(output_dir):
        os.rmdir(output_dir)


def test_load_models(audio_bulk, model_config):
    (
        model_name,
        processor_name,
        model_class,
        processor_class,
        use_cuda,
        precision,
        quantization,
        device_map,
        max_memory,
        torchscript,
    ) = model_config

    model, processor = audio_bulk.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        use_cuda=use_cuda,
        precision=precision,
        quantization=quantization,
        device_map=device_map,
        max_memory=max_memory,
        torchscript=torchscript,
    )
    assert model is not None
    assert processor is not None
    assert len(list(model.named_modules())) > 0

    del model
    del processor
    torch.cuda.empty_cache()


def test_process_audio(audio_bulk, model_config):
    (
        model_name,
        processor_name,
        model_class,
        processor_class,
        use_cuda,
        precision,
        quantization,
        device_map,
        max_memory,
        torchscript,
    ) = model_config

    audio_bulk.model, audio_bulk.processor = audio_bulk.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        use_cuda=use_cuda,
        precision=precision,
        quantization=quantization,
        device_map=device_map,
        max_memory=max_memory,
        torchscript=torchscript,
    )

    audio_input = b"mock_audio_data"
    model_sampling_rate = 16000

    result = audio_bulk.process_wav2vec2(audio_input, model_sampling_rate, {}, 0, 0)
    assert result["transcription"] is not None
    assert isinstance(result["transcription"], str)
    assert result["segments"] is not None
    assert isinstance(result["segments"], list)

    del audio_bulk.model
    del audio_bulk.processor
    torch.cuda.empty_cache()
