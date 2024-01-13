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

import base64
from typing import List, Optional

import cherrypy
from geniusrise import Spout, State, StreamingOutput


class Webhook(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the Webhook class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius Webhook rise \
            streaming \
                --output_kafka_topic webhook_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args endpoint=* port=3000
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_webhook_spout:
                name: "Webhook"
                method: "listen"
                args:
                    endpoint: "*"
                    port: 3000
                output:
                    type: "streaming"
                    args:
                        output_topic: "webhook_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.buffer: List[dict] = []

    def _check_auth(self, username, password):
        auth_header = cherrypy.request.headers.get("Authorization")
        if auth_header:
            auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            provided_username, provided_password = auth_decoded.split(":", 1)
            if provided_username != username or provided_password != password:
                raise cherrypy.HTTPError(401, "Unauthorized")
        else:
            raise cherrypy.HTTPError(401, "Unauthorized")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def default(self, username=None, password=None):
        if username and password:
            self._check_auth(username, password)

        try:
            data = cherrypy.request.json

            # Add additional data about the endpoint and headers
            enriched_data = {
                "data": data,
                "endpoint": cherrypy.url(),
                "headers": dict(cherrypy.request.headers),
            }

            # Use the output's save method
            self.output.save(enriched_data)

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            if "success_count" not in current_state.keys():
                current_state = {"success_count": 0, "failure_count": 0}
            current_state["success_count"] += 1
            self.state.set_state(self.id, current_state)

            return ""
        except Exception as e:
            self.log.error(f"Error processing webhook data: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

            cherrypy.response.status = 500
            return "Error processing data"

    def listen(
        self,
        endpoint: str = "*",
        port: int = 3000,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        📖 Start listening for data from the webhook.

        Args:
            endpoint (str): The webhook endpoint to listen to. Defaults to "*".
            port (int): The port to listen on. Defaults to 3000.
            username (Optional[str]): The username for basic authentication. Defaults to None.
            password (Optional[str]): The password for basic authentication. Defaults to None.

        Raises:
            Exception: If unable to start the CherryPy server.
        """
        # Disable CherryPy's default loggers
        cherrypy.log.access_log.propagate = False
        cherrypy.log.error_log.propagate = False

        # Set CherryPy's error and access loggers to use your logger
        cherrypy.log.error_log.addHandler(self.log)
        cherrypy.log.access_log.addHandler(self.log)

        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,  # Disable logging to the console
            }
        )
        cherrypy.tree.mount(self, "/")
        cherrypy.engine.start()
        cherrypy.engine.block()
