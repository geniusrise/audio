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
from typing import Optional

import redis  # type: ignore
from geniusrise import Spout, State, StreamingOutput


class RedisStream(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the RedisStream class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius RedisStream rise \
            streaming \
                --output_kafka_topic redis_stream_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args stream_key=my_stream host=localhost port=6379 db=0
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_redis_stream:
                name: "RedisStream"
                method: "listen"
                args:
                    stream_key: "my_stream"
                    host: "localhost"
                    port: 6379
                    db: 0
                output:
                    type: "streaming"
                    args:
                        output_topic: "redis_stream_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    async def _listen(
        self,
        stream_key: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        last_id: Optional[str] = None,
    ):
        """
        📖 Start listening for data from the Redis stream.

        Args:
            stream_key (str): The Redis stream key to listen to.
            host (str): The Redis server host. Defaults to "localhost".
            port (int): The Redis server port. Defaults to 6379.
            db (int): The Redis database index. Defaults to 0.
            password (Optional[str]): The password for authentication. Defaults to None.
            last_id (Optional[str]): The last message ID that was processed. Defaults to None.

        Raises:
            Exception: If unable to connect to the Redis server.
        """
        try:
            self.log.info(f"Starting to listen to Redis stream {stream_key} on host {host}")

            self.redis = redis.StrictRedis(host=host, port=port, password=password, decode_responses=True, db=db)
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
                "last_id": last_id,
            }
            last_id = (
                current_state["last_id"]
                if "last_id" in current_state and last_id is None and current_state["last_id"] is not None
                else "0"
                if last_id is None
                else last_id
            )

            while True:
                try:
                    # Use run_in_executor to run the synchronous redis call in a separate thread
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis.xread,
                        {stream_key: last_id, "count": 10, "block": 1000},
                    )

                    for _, messages in result:
                        for msg_id, fields in messages:
                            last_id = msg_id

                            # Enrich the data with metadata about the stream key and message ID
                            enriched_data = {
                                "data": fields,
                                "stream_key": stream_key,
                                "message_id": msg_id,
                            }

                            # Use the output's save method
                            self.output.save(enriched_data)

                            # Update the state using the state
                            current_state = self.state.get_state(self.id) or {
                                "success_count": 0,
                                "failure_count": 0,
                                "last_id": last_id,
                            }
                            current_state["success_count"] += 1
                            current_state["last_id"] = last_id
                            self.state.set_state(self.id, current_state)
                except Exception as e:
                    self.log.exception(f"Failed to process SNS message: {e}")
                    current_state["failure_count"] += 1
                    self.state.set_state(self.id, current_state)

                await asyncio.sleep(1)  # to prevent high CPU usage

        except Exception as e:
            self.log.error(f"Error processing Redis Stream message: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
                "last_id": last_id,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

    def listen(
        self,
        stream_key: str,
        host: str = "localhost",
        port: int = 6379,
        db=0,
        password: Optional[str] = None,
    ):
        """
        📖 Start the asyncio event loop to listen for data from the Redis stream.

        Args:
            stream_key (str): The Redis stream key to listen to.
            host (str): The Redis server host. Defaults to "localhost".
            port (int): The Redis server port. Defaults to 6379.
            db (int): The Redis database index. Defaults to 0.
            password (Optional[str]): The password for authentication. Defaults to None.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._listen(stream_key=stream_key, host=host, port=port, db=db, password=password))
