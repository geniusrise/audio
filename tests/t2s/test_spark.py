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

# test_t2s_spark.py

import os
import tempfile

import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyspark.sql import SparkSession

from geniusrise_audio.t2s.spark import TextToSpeechSpark


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.appName("TextToSpeechSparkTest").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def t2s_spark(spark):
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    t2s_spark = TextToSpeechSpark(
        input=input,
        output=output,
        state=state,
        spark_session=spark,
        model_name="facebook/fastspeech2-en-ljspeech",
        model_class="SpeechSynthesisModel",
        processor_class="SpeechSynthesisProcessor",
    )
    yield t2s_spark

    os.rmdir(input_dir)
    os.rmdir(output_dir)


def test_synthesize_dataframe(t2s_spark):
    data = [
        ("Test text 1", "0"),
        ("Test text 2", "1"),
    ]
    columns = ["text", "voice_preset"]
    df = t2s_spark.spark.createDataFrame(data, columns)

    t2s_spark.load_models = lambda *args, **kwargs: (None, None)
    t2s_spark.process_text = lambda *args, **kwargs: b"Test audio data"

    result_df = t2s_spark.synthesize_dataframe(df, text_col="text", voice_preset_col="voice_preset")

    assert result_df.count() == 2
    assert "text" in result_df.columns
    assert "voice_preset" in result_df.columns
    assert "audio" in result_df.columns
    result_df.show()
    audio_data = result_df.select("audio").collect()
    assert audio_data[0]["audio"] == "Test audio data"
    assert audio_data[1]["audio"] == "Test audio data"
