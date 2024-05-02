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

# test_s2t_spark.py

import os
import tempfile

import pytest
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from pyspark.sql import SparkSession

from geniusrise_audio.s2t.spark import SpeechToTextSpark


@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder.appName("SpeechToTextSparkTest").getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def s2t_spark(spark):
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "api_input")
    output = BatchOutput(output_dir, "geniusrise-test", "api_output")
    state = InMemoryState()

    s2t_spark = SpeechToTextSpark(
        input=input,
        output=output,
        state=state,
        spark_session=spark,
        model_name="facebook/wav2vec2-base-960h",
        model_class="Wav2Vec2ForCTC",
        processor_class="Wav2Vec2Processor",
    )
    yield s2t_spark

    os.rmdir(input_dir)
    os.rmdir(output_dir)


def test_transcribe_dataframe(s2t_spark):
    data = [
        (b"mock_audio_data_1",),
        (b"mock_audio_data_2",),
    ]
    columns = ["audio"]
    df = s2t_spark.spark.createDataFrame(data, columns)

    s2t_spark.load_models = lambda *args, **kwargs: (None, None)
    s2t_spark.process_audio = lambda *args, **kwargs: "Test transcription"

    result_df = s2t_spark.transcribe_dataframe(df, audio_col="audio")

    assert result_df.count() == 2
    assert "audio" in result_df.columns
    assert "transcription" in result_df.columns
    result_df.show()
    transcriptions = result_df.select("transcription").collect()
    assert transcriptions[0]["transcription"] == "Test transcription"
    assert transcriptions[1]["transcription"] == "Test transcription"
