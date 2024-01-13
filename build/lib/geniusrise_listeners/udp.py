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

import socket

from geniusrise import Spout, State, StreamingOutput


class Udp(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the Udp class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius Udp rise \
            streaming \
                --output_kafka_topic udp_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args host=localhost port=12345
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_udp_spout:
                name: "Udp"
                method: "listen"
                args:
                    host: "localhost"
                    port: 12345
                output:
                    type: "streaming"
                    args:
                        output_topic: "udp_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    def listen(self, host: str = "localhost", port: int = 12345):
        """
        📖 Start listening for data from the UDP server.

        Args:
            host (str): The UDP server host. Defaults to "localhost".
            port (int): The UDP server port. Defaults to 12345.

        Raises:
            Exception: If unable to connect to the UDP server.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((host, port))
            while True:
                try:
                    data, addr = s.recvfrom(1024)

                    # Enrich the data with metadata about the sender's address and port
                    enriched_data = {
                        "data": data.decode("utf-8"),  # Assuming the data is a utf-8 encoded string
                        "sender_address": addr[0],
                        "sender_port": addr[1],
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
                    self.log.error(f"Error processing UDP data: {e}")

                    # Update the state using the state
                    current_state = self.state.get_state(self.id) or {
                        "success_count": 0,
                        "failure_count": 0,
                    }
                    current_state["failure_count"] += 1
                    self.state.set_state(self.id, current_state)
