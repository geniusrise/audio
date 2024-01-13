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

import boto3
from botocore.exceptions import ClientError
from geniusrise import Spout, State, StreamingOutput


class SNS(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the SNS class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius SNS rise \
            streaming \
                --output_kafka_topic sns_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_sns_spout:
                name: "SNS"
                method: "listen"
                output:
                    type: "streaming"
                    args:
                        output_topic: "sns_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs
        self.sns = boto3.resource("sns")

    async def _listen_to_subscription(self, subscription):
        """
        📖 Listen to a specific subscription.

        Args:
            subscription: The subscription to listen to.

        Raises:
            ClientError: If unable to connect to the AWS SNS service.
        """
        try:
            while True:
                try:
                    messages = subscription.get_messages()
                    for message in messages:
                        # Enrich the data with metadata about the subscription ARN
                        enriched_data = {
                            "data": message,
                            "subscription_arn": subscription.arn,
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
                    self.log.exception(f"Failed to process SNS message: {e}")
                    current_state = self.state.get_state(self.id) or {
                        "success_count": 0,
                        "failure_count": 0,
                    }
                    current_state["failure_count"] += 1
                    self.state.set_state(self.id, current_state)
        except ClientError as e:
            self.log.error(f"Error processing SNS message from subscription {subscription.arn}: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

    async def _listen(self):
        """
        📖 Start listening for data from AWS SNS.

        Raises:
            ClientError: If unable to connect to the AWS SNS service.
        """
        try:
            for topic in self.sns.topics.all():
                for subscription in topic.subscriptions.all():
                    self.log.info(f"Listening to topic {topic.arn} with subscription {subscription.arn}")
                    await self._listen_to_subscription(subscription)
        except ClientError as e:
            self.log.error(f"Error listening to AWS SNS: {e}")

            # Update the state using the state
            current_state = self.state.get_state(self.id) or {
                "success_count": 0,
                "failure_count": 0,
            }
            current_state["failure_count"] += 1
            self.state.set_state(self.id, current_state)

    def listen(self):
        """
        📖 Start the asyncio event loop to listen for data from AWS SNS.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._listen())
        self.log.info("Exiting...")
