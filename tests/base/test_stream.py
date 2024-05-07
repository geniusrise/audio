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

import tempfile

import pytest
from geniusrise.core import StreamingInput, StreamingOutput, InMemoryState

from geniusrise_audio import AudioStream


@pytest.fixture(scope="module")
def audio_stream():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = StreamingInput(input_topic="test-input-topic", kafka_cluster_connection_string="localhost:9092")
    output = StreamingOutput(output_topic="test-output-topic", kafka_cluster_connection_string="localhost:9092")
    state = InMemoryState(1)

    audio_stream = AudioStream(
        input=input,
        output=output,
        state=state,
    )
    yield audio_stream


@pytest.mark.parametrize(
    "model_name, processor_name, model_class, processor_class, use_cuda, precision, quantization, device_map, torchscript, compile, better_transformers, use_whisper_cpp, use_faster_whisper",
    [
        # fmt: off
        ("facebook/wav2vec2-base-960h", "facebook/wav2vec2-base-960h", "Wav2Vec2ForCTC", "Wav2Vec2Processor", True, "float32", 0, "cuda:0", False, False, False, False, False),
        ("openai/whisper-small", "openai/whisper-small", "WhisperForConditionalGeneration", "AutoProcessor", False, "float32", 4, None, False, False, False, False, False),
        ("openai/whisper-medium", "openai/whisper-medium", "WhisperForConditionalGeneration", "AutoProcessor", True, "float16", 0, "cuda:0", False, True, False, False, False),
        ("large", "large", "WhisperForConditionalGeneration", "AutoProcessor", True, "bfloat16", 0, "cuda:0", False, False, True, False, False),
        ("large", "large", "WhisperForConditionalGeneration", "AutoProcessor", True, "bfloat16", 0, "cuda:0", False, False, False, True, False),
        ("large-v3", "large-v3", "WhisperForConditionalGeneration", "AutoProcessor", None, "float32", 0, "cuda:0", None, None, False, False, True),
        # fmt: on
    ],
)
def test_load_models(
    audio_stream,
    model_name,
    processor_name,
    model_class,
    processor_class,
    use_cuda,
    precision,
    quantization,
    device_map,
    torchscript,
    compile,
    better_transformers,
    use_whisper_cpp,
    use_faster_whisper,
):
    model, processor = audio_stream.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        use_cuda=use_cuda,
        precision=precision,
        quantization=quantization,
        device_map=device_map,
        torchscript=torchscript,
        compile=compile,
        better_transformers=better_transformers,
        use_whisper_cpp=use_whisper_cpp,
        use_faster_whisper=use_faster_whisper,
    )

    assert model is not None
    if not use_whisper_cpp and not use_faster_whisper:
        assert processor is not None
    else:
        assert processor is None
