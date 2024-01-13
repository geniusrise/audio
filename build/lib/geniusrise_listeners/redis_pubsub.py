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

import redis  # type: ignore
from geniusrise import Spout, State, StreamingOutput


class RedisPubSub(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the RedisPubSub class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius RedisPubSub rise \
            streaming \
                --output_kafka_topic redis_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args channel=my_channel host=localhost port=6379 db=0
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_redis_spout:
                name: "RedisPubSub"
                method: "listen"
                args:
                    channel: "my_channel"
                    host: "localhost"
                    port: 6379
                    db: 0
                output:
                    type: "streaming"
                    args:
                        output_topic: "redis_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs

    def listen(
        self,
        channel: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        """
        📖 Start listening for data from the Redis Pub/Sub channel.

        Args:
            channel (str): The Redis Pub/Sub channel to listen to.
            host (str): The Redis server host. Defaults to "localhost".
            port (int): The Redis server port. Defaults to 6379.
            db (int): The Redis database index. Defaults to 0.
            password (Optional[str]): The password for authentication. Defaults to None.

        Raises:
            Exception: If unable to connect to the Redis server.
        """
        self.redis = redis.StrictRedis(host=host, port=port, password=password, decode_responses=True, db=db)
        pubsub = self.redis.pubsub()
        pubsub.subscribe(channel)

        self.log.info(f"Listening to channel {channel} on Redis server at {host}:{port}")

        for message in pubsub.listen():
            try:
                if message["type"] == "message":
                    data = json.loads(message["data"])

                    # Enrich the data with metadata about the channel
                    enriched_data = {
                        "data": data,
                        "channel": channel,
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
                self.log.error(f"Error processing Redis Pub/Sub message: {e}")

                # Update the state using the state
                current_state = self.state.get_state(self.id) or {
                    "success_count": 0,
                    "failure_count": 0,
                }
                current_state["failure_count"] += 1
                self.state.set_state(self.id, current_state)
