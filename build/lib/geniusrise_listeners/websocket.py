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

import asyncio

import websockets
from geniusrise import Spout, State, StreamingOutput


class Websocket(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the Websocket class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius Websocket rise \
            streaming \
                --output_kafka_topic websocket_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args host=localhost port=8765
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_websocket_spout:
                name: "Websocket"
                method: "listen"
                args:
                    host: "localhost"
                    port: 8765
                output:
                    type: "streaming"
                    args:
                        output_topic: "websocket_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    async def __listen(self, host: str, port: int):
        """
        Start listening for data from the WebSocket server.
        """
        async with websockets.serve(self.receive_message, host, port):  # type: ignore
            await asyncio.Future()  # run forever

    async def receive_message(self, websocket, path):
        """
        Receive a message from a WebSocket client and save it along with metadata.

        Args:
            websocket: WebSocket client connection.
            path: WebSocket path.
        """
        try:
            data = await websocket.recv()

            # Add additional metadata
            enriched_data = {
                "data": data,
                "path": path,
                "client_address": websocket.remote_address,
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
            self.log.error(f"Error processing WebSocket data: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

    def listen(self, host: str = "localhost", port: int = 8765):
        """
        📖 Start the WebSocket server.

        Args:
            host (str): The WebSocket server host. Defaults to "localhost".
            port (int): The WebSocket server port. Defaults to 8765.

        Raises:
            Exception: If unable to start the WebSocket server.
        """
        asyncio.run(self.__listen(host, port))
