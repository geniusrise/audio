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

from typing import Any, Dict, List, Optional
import json
import os
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_audio.base import AudioBulk, send_email
from transformers import AutoModelForCTC, AutoProcessor
from datasets import Dataset, load_from_disk
import pandas as pd
import yaml  # type: ignore
import glob
import sqlite3
import xml.etree.ElementTree as ET
from pyarrow import feather, parquet as pq
import torchaudio
from torch import Tensor


class TextToSpeechBulk(AudioBulk):
    r"""
    TextToSpeechBulk is designed for bulk processing of text-to-speech tasks. It utilizes a range of models from Hugging Face,
    converting text inputs to speech audio outputs.

    Attributes:
        model (AutoModelForCTC): The text-to-speech model.
        processor (AutoProcessor): The processor to prepare input text data for the model.

    Methods:
        synthesize_speech(texts: List[str], **kwargs: Any) -> None:
            Synthesizes speech from a batch of text inputs.

    Example CLI Usage:
    ```bash
    genius TextToSpeechBulk rise \
        batch \
            --input_s3_bucket geniusrise-test \
            --input_s3_folder input/summz \
        batch \
            --output_s3_bucket geniusrise-test \
            --output_s3_folder output/summz \
        postgres \
            --postgres_host 127.0.0.1 \
            --postgres_port 5432 \
            --postgres_user postgres \
            --postgres_password postgres \
            --postgres_database geniusrise\
            --postgres_table state \
        --id facebook/bart-large-cnn-lol \
        summarize \
            --args \
                model_name="facebook/bart-large-cnn" \
                model_class="AutoModelForSeq2SeqLM" \
                tokenizer_class="AutoTokenizer" \
                use_cuda=True \
                precision="float" \
                quantization=0 \
                device_map="cuda:0" \
                max_memory=None \
                torchscript=False \
                generation_bos_token_id=0 \
                generation_decoder_start_token_id=2 \
                generation_early_stopping=true \
                generation_eos_token_id=2 \
                generation_forced_bos_token_id=0 \
                generation_forced_eos_token_id=2 \
                generation_length_penalty=2.0 \
                generation_max_length=142 \
                generation_min_length=56 \
                generation_no_repeat_ngram_size=3 \
                generation_num_beams=4 \
                generation_pad_token_id=1 \
                generation_do_sample=false
    ```
    """

    model: AutoModelForCTC
    processor: AutoProcessor

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the TextToSpeechBulk with configurations for text-to-speech processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)

    def load_dataset(self, dataset_path: str, max_length: int = 512, **kwargs) -> Optional[Dataset]:
        r"""
        Loads and configures the specified text-to-speech model and processor.

        Args:
            dataset_path (str): The path to the dataset directory.
            max_length (int, optional): The maximum length for tokenization. Defaults to 512.
            **kwargs: Additional keyword arguments to pass to the underlying dataset loading functions.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            Exception: If there was an error loading the dataset.

        ## Supported Data Formats and Structures:

        ### Dataset files saved by Hugging Face datasets library
        The directory should contain 'dataset_info.json' and other related files.

        ### JSONL
        Each line is a JSON object representing an example.
        ```json
        {"text": "The text content"}
        ```

        ### CSV
        Should contain 'text' column.
        ```csv
        text
        "The text content"
        ```

        ### Parquet
        Should contain 'text' column.

        ### JSON
        An array of dictionaries with 'text' key.
        ```json
        [{"text": "The text content"}]
        ```

        ### XML
        Each 'record' element should contain 'text' child element.
        ```xml
        <record>
            <text>The text content</text>
        </record>
        ```

        ### YAML
        Each document should be a dictionary with 'text' key.
        ```yaml
        - text: "The text content"
        ```

        ### TSV
        Should contain 'text' column separated by tabs.

        ### Excel (.xls, .xlsx)
        Should contain 'text' column.

        ### SQLite (.db)
        Should contain a table with 'text' column.

        ### Feather
        Should contain 'text' column.
        """

        self.max_length = max_length

        self.label_to_id = self.model.config.label2id if self.model and self.model.config.label2id else {}  # type: ignore

        try:
            self.log.info(f"Loading dataset from {dataset_path}")
            if os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
                # Load dataset saved by Hugging Face datasets library
                return load_from_disk(dataset_path)
            else:
                data = []
                for filename in glob.glob(f"{dataset_path}/**/*", recursive=True):
                    filepath = os.path.join(dataset_path, filename)
                    if filename.endswith(".jsonl"):
                        with open(filepath, "r") as f:
                            for line in f:
                                example = json.loads(line)
                                data.append(example)

                    elif filename.endswith(".csv"):
                        df = pd.read_csv(filepath)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".parquet"):
                        df = pq.read_table(filepath).to_pandas()
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".json"):
                        with open(filepath, "r") as f:
                            json_data = json.load(f)
                            data.extend(json_data)

                    elif filename.endswith(".xml"):
                        tree = ET.parse(filepath)
                        root = tree.getroot()
                        for record in root.findall("record"):
                            text = record.find("text").text  # type: ignore
                            data.append({"text": text})

                    elif filename.endswith(".yaml") or filename.endswith(".yml"):
                        with open(filepath, "r") as f:
                            yaml_data = yaml.safe_load(f)
                            data.extend(yaml_data)

                    elif filename.endswith(".tsv"):
                        df = pd.read_csv(filepath, sep="\t")
                        data.extend(df.to_dict("records"))

                    elif filename.endswith((".xls", ".xlsx")):
                        df = pd.read_excel(filepath)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".db"):
                        conn = sqlite3.connect(filepath)
                        query = "SELECT text FROM dataset_table;"
                        df = pd.read_sql_query(query, conn)
                        data.extend(df.to_dict("records"))

                    elif filename.endswith(".feather"):
                        df = feather.read_feather(filepath)
                        data.extend(df.to_dict("records"))

                if hasattr(self, "map_data") and self.map_data:
                    fn = eval(self.map_data)  # type: ignore
                    data = [fn(d) for d in data]
                else:
                    data = data

                return Dataset.from_pandas(pd.DataFrame(data))
        except Exception as e:
            self.log.exception(f"Error occurred when loading dataset from {dataset_path}. Error: {e}")
            raise

    def synthesize_speech(
        self,
        model_name: str,
        model_class: str = "AutoModelForCausalLM",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = False,
        compile: bool = True,
        batch_size: int = 8,
        notification_email: Optional[str] = None,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """
        Synthesizes speech from a batch of text inputs using the text-to-speech model.
        Args:
            model_name (str): Name of the model to be used.
            model_class (str): Class name of the model (default "AutoModelForCausalLM").
            processor_class (str): Class name of the processor (default "AutoProcessor").
            use_cuda (bool): Whether to use CUDA for model inference (default False).
            precision (str): Precision for model computation (default "float16").
            quantization (int): Level of quantization for optimizing model size and speed (default 0).
            device_map (Union[str, Dict, None]): Specific device to use for computation (default "auto").
            max_memory (Dict): Maximum memory configuration for devices.
            torchscript (bool): Whether to use a TorchScript-optimized version of the model. Defaults to False.
            compile (bool): Whether to compile the model before fine-tuning. Defaults to True.
            batch_size (int): Number of transcriptions to process simultaneously (default 8).
            notification_email (Optional[str]): Email address for notifications.
            max_length: (int): Maximum length of the input after which to truncate.
            **kwargs: Arbitrary keyword arguments for model and generation configurations.
        """
        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            processor_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            processor_name = model_name
        else:
            model_revision = None
            processor_revision = None
            processor_name = model_name

        self.model_name = model_name
        self.processor_name = processor_name
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.notification_email = notification_email
        self.compile = compile
        self.max_length = max_length

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

        generation_args = {k.replace("generation_", ""): v for k, v in kwargs.items() if "generation_" in k}
        self.generation_args = generation_args

        # Load models and processor
        self.model, self.processor = self.load_models(
            model_name=model_name,
            processor_name=processor_name,
            model_revision=model_revision,
            processor_revision=processor_revision,
            model_class=model_class,
            processor_class=processor_class,
            use_cuda=use_cuda,
            precision=precision,
            quantization=quantization,
            device_map=device_map,
            max_memory=max_memory,
            torchscript=torchscript,
            compile=compile,
            **self.model_args,
        )

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        _dataset = self.load_dataset(dataset_path)
        if _dataset is None:
            self.log.error("Failed to load dataset.")
            return
        dataset = _dataset["text"]

        # Process the batch of texts
        for i in range(0, len(dataset), batch_size):
            batch_texts = dataset[i : i + batch_size]
            self._process_and_save_batch(batch_texts, i)

        # Finalize
        if notification_email:
            self.output.flush()
            # Function to send email (to be implemented)
            send_email(recipient=notification_email, bucket_name=self.output.bucket, prefix=self.output.s3_folder)

    def _process_and_save_batch(self, batch_texts: List[str], batch_idx: int) -> None:
        """
        Processes a batch of texts and saves the synthesized speech.

        Args:
            batch_texts (List[str]): The batch of texts to synthesize.
            batch_idx (int): The batch index.
        """
        # Synthesize speech
        inputs = self.processor(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        outputs = self.model.generate(**inputs)

        # Save synthesized speech
        for idx, speech in enumerate(outputs):
            file_name = f"synthesized_speech_{batch_idx}_{idx}.wav"
            file_path = os.path.join(self.output.output_folder, file_name)
            self.save_speech_to_wav(speech, file_path)

    def save_speech_to_wav(self, speech: Tensor, file_path: str) -> None:
        """
        Saves synthesized speech tensor to a WAV file.

        Args:
            speech (Tensor): The speech tensor output from the TTS model.
            file_path (str): The file path where the WAV file will be saved.
        """
        # Assuming the speech tensor is in the format expected by torchaudio
        torchaudio.save(file_path, speech.cpu(), sample_rate=self.model.config.sampling_rate)
