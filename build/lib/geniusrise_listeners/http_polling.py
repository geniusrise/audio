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

import time
from typing import Dict, Optional

import requests
from geniusrise import Spout, State, StreamingOutput
from requests.exceptions import HTTPError, RequestException


class RESTAPIPoll(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the RESTAPIPoll class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius RESTAPIPoll rise \
            streaming \
                --output_kafka_topic restapi_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args url=https://api.example.com method=GET interval=60
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_restapi_poll:
                name: "RESTAPIPoll"
                method: "listen"
                args:
                    url: "https://api.example.com"
                    method: "GET"
                    interval: 60
                output:
                    type: "streaming"
                    args:
                        output_topic: "restapi_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    def poll_api(
        self,
        url: str,
        method: str,
        body: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ):
        """
        📖 Start polling the REST API for data.

        Args:
            url (str): The API endpoint.
            method (str): The HTTP method (GET, POST, etc.).
            interval (int): The polling interval in seconds.
            body (Optional[Dict]): The request body. Defaults to None.
            headers (Optional[Dict[str, str]]): The request headers. Defaults to None.
            params (Optional[Dict[str, str]]): The request query parameters. Defaults to None.

        Raises:
            Exception: If unable to connect to the REST API server.
        """
        try:
            response = getattr(requests, method.lower())(url, json=body, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Add additional data about the request
            enriched_data = {
                "data": data,
                "url": url,
                "method": method,
                "headers": headers,
                "params": params,
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
        except HTTPError:
            self.log.error(
                f"HTTP error {response.status_code} when fetching data from {url}. Response: {response.text}"
            )
        except RequestException as e:
            self.log.error(f"Error fetching data from REST API: {e}")
        except Exception as e:
            self.log.error(f"Unexpected error: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

    def listen(
        self,
        url: str,
        method: str,
        interval: int = 60,
        body: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
    ):
        """
        Start polling the REST API for data.

        Args:
            url (str): The API endpoint.
            method (str): The HTTP method (GET, POST, etc.).
            interval (int): The polling interval in seconds. Defaults to 60.
            body (Optional[Dict]): The request body. Defaults to None.
            headers (Optional[Dict[str, str]]): The request headers. Defaults to None.
            params (Optional[Dict[str, str]]): The request query parameters. Defaults to None.
        """

        while True:
            self.poll_api(url, method, body, headers, params)
            time.sleep(interval)
