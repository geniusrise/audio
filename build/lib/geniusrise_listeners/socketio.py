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

import socketio
from typing import Optional
from geniusrise import Spout, State, StreamingOutput


class SocketIo(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the SocketIo class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius SocketIo rise \
            streaming \
                --output_kafka_topic socketio_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args url=http://localhost:3000 namespace=/chat
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_socketio_spout:
                name: "SocketIo"
                method: "listen"
                args:
                    url: "http://localhost:3000"
                    namespace: "/chat"
                output:
                    type: "streaming"
                    args:
                        output_topic: "socketio_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs
        self.sio = socketio.Client()

    def listen(
        self,
        url: str,
        namespace: Optional[str] = None,
        event: str = "message",
        auth: Optional[dict] = None,
    ):
        """
        📖 Start listening for data from the Socket.io server.

        Args:
            url (str): The Socket.io server URL.
            namespace (Optional[str]): The Socket.io namespace. Defaults to None.
            event (str): The Socket.io event to listen to. Defaults to "message".
            auth (Optional[dict]): Authentication dictionary. Defaults to None.

        Raises:
            Exception: If unable to connect to the Socket.io server.
        """
        try:
            self.log.info(f"Connecting to Socket.io server at {url}")

            if auth:
                self.sio.connect(url, auth=auth, namespaces=[namespace] if namespace else None)
            else:
                self.sio.connect(url, namespaces=[namespace] if namespace else None)

            @self.sio.on(event, namespace=namespace)
            def handle_message(message):
                self._message_handler(message)

            self.log.info(f"Listening for Socket.io messages on event '{event}'")
        except Exception as e:
            self.log.error(f"Error connecting to Socket.io server: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)
