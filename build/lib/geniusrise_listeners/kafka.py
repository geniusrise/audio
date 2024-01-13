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

import json
from typing import Optional

from confluent_kafka import Consumer, KafkaError
from geniusrise import Spout, State, StreamingOutput


class Kafka(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the Kafka class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius Kafka rise \
            streaming \
                --output_kafka_topic kafka_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args topic=my_topic group_id=my_group
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_kafka_spout:
                name: "Kafka"
                method: "listen"
                args:
                    topic: "my_topic"
                    group_id: "my_group"
                output:
                    type: "streaming"
                    args:
                        output_topic: "kafka_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    def listen(
        self,
        topic: str,
        group_id: str,
        bootstrap_servers: str = "localhost:9092",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        📖 Start listening for data from the Kafka topic.

        Args:
            topic (str): The Kafka topic to listen to.
            group_id (str): The Kafka consumer group ID.
            bootstrap_servers (str): The Kafka bootstrap servers. Defaults to "localhost:9092".
            username (Optional[str]): The username for SASL/PLAIN authentication. Defaults to None.
            password (Optional[str]): The password for SASL/PLAIN authentication. Defaults to None.

        Raises:
            Exception: If unable to connect to the Kafka server.
        """
        config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
        if username and password:
            config.update(
                {
                    "security.protocol": "SASL_PLAINTEXT",
                    "sasl.mechanisms": "PLAIN",
                    "sasl.username": username,
                    "sasl.password": password,
                }
            )
        consumer = Consumer(config)

        consumer.subscribe([topic])

        while True:
            try:
                message = consumer.poll(1.0)

                if message is None:
                    continue
                if message.error():
                    if message.error().code() == KafkaError._PARTITION_EOF:
                        self.log.info(f"Reached end of topic {topic}, partition {message.partition()}")
                    else:
                        self.log.error(f"Error while consuming message: {message.error()}")
                else:
                    # Use the output's save method
                    self.output.save(json.loads(message.value()))

                    # Update the state using the state
                    current_state = self.state.get_state(self.id) or {
                        "success_count": 0,
                        "failure_count": 0,
                    }
                    current_state["success_count"] += 1
                    self.state.set_state(self.id, current_state)
            except Exception as e:
                self.log.error(f"Error processing Kafka message: {e}")

                # Update the state using the state
                current_state = self.state.get_state(self.id) or {
                    "success_count": 0,
                    "failure_count": 0,
                }
                current_state["failure_count"] += 1
                self.state.set_state(self.id, current_state)

        consumer.close()
