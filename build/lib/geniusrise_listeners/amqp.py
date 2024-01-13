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

import pika
from geniusrise import Spout, State, StreamingOutput


class RabbitMQ(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the RabbitMQ class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius RabbitMQ rise \
            streaming \
                --output_kafka_topic rabbitmq_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args queue_name=my_queue host=localhost
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_rabbitmq_spout:
                name: "RabbitMQ"
                method: "listen"
                args:
                    queue_name: "my_queue"
                    host: "localhost"
                output:
                    type: "streaming"
                    args:
                        output_topic: "rabbitmq_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    def _callback(self, ch, method, properties, body):
        """
        Callback function that is called when a message is received.

        Args:
            ch: Channel.
            method: Method.
            properties: Properties.
            body: Message body.
        """
        try:
            data = json.loads(body)

            # Enrich the data with metadata about the method and properties
            enriched_data = {
                "data": data,
                "method": method.routing_key,
                "properties": dict(properties.headers),
            }

            # Use the output's save method
            self.output.save(enriched_data)

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["success_count"] += 1
            self.state.set_state(self.id, current_state)
        except Exception as e:
            self.log.error(f"Error processing RabbitMQ message: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

    def listen(
        self,
        queue_name: str,
        host: str = "localhost",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        📖 Start listening for data from the RabbitMQ server.

        Args:
            queue_name (str): The RabbitMQ queue name to listen to.
            host (str): The RabbitMQ server host. Defaults to "localhost".
            username (Optional[str]): The username for authentication. Defaults to None.
            password (Optional[str]): The password for authentication. Defaults to None.

        Raises:
            Exception: If unable to connect to the RabbitMQ server.
        """
        try:
            self.log.info("Starting RabbitMQ listener...")
            credentials = pika.PlainCredentials(username, password) if username and password else None
            connection = (
                pika.BlockingConnection(pika.ConnectionParameters(host=host, credentials=credentials))
                if credentials
                else pika.BlockingConnection(pika.ConnectionParameters(host=host))
            )
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_consume(queue=queue_name, on_message_callback=self._callback, auto_ack=True)
            self.log.info("Waiting for messages. To exit press CTRL+C")
            channel.start_consuming()
        except Exception as e:
            self.log.error(f"Error listening to RabbitMQ: {e}")
            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)
