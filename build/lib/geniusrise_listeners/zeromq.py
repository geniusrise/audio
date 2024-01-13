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
import zmq
from geniusrise import Spout, State, StreamingOutput
from typing import Optional


class ZeroMQ(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the ZeroMQ class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius ZeroMQ rise \
            streaming \
                --output_kafka_topic zmq_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args endpoint=tcp://localhost:5555 topic=my_topic syntax=json
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_zmq_spout:
                name: "ZeroMQ"
                method: "listen"
                args:
                    endpoint: "tcp://localhost:5555"
                    topic: "my_topic"
                    syntax: "json"
                output:
                    type: "streaming"
                    args:
                        output_topic: "zmq_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    def listen(
        self,
        endpoint: str,
        topic: str,
        syntax: str,
        socket_type: Optional[str] = "SUB",
    ):
        """
        📖 Start listening for data from the ZeroMQ server.

        Args:
            endpoint (str): The endpoint to connect to (e.g., "tcp://localhost:5555").
            topic (str): The topic to subscribe to.
            syntax (str): The syntax to be used (e.g., "json").
            socket_type (Optional[str]): The type of ZeroMQ socket (default is "SUB").

        Raises:
            Exception: If unable to connect to the ZeroMQ server or process messages.
        """
        context = zmq.Context()

        # Create a socket of the specified type
        if socket_type == "SUB":
            socket = context.socket(zmq.SUB)
            socket.connect(endpoint)
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        else:
            raise ValueError(f"Unsupported socket type: {socket_type}")

        try:
            while True:
                # Receive the message
                message = socket.recv_string()

                # Parse the message based on the syntax
                if syntax == "json":
                    data = json.loads(message.split(" ", 1)[1])
                else:
                    data = message

                # Enrich the data with metadata about the topic and syntax
                enriched_data = {
                    "data": data,
                    "topic": topic,
                    "syntax": syntax,
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
            self.log.error(f"Error processing ZeroMQ message: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)
