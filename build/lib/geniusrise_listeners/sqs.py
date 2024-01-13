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

import boto3
from geniusrise import Spout, State, StreamingOutput


class SQS(Spout):
    def __init__(self, output: StreamingOutput, state: State, **kwargs):
        r"""
        Initialize the SQS class.

        Args:
            output (StreamingOutput): An instance of the StreamingOutput class for saving the data.
            state (State): An instance of the State class for maintaining the state.
            **kwargs: Additional keyword arguments.

        ## Using geniusrise to invoke via command line
        ```bash
        genius SQS rise \
            streaming \
                --output_kafka_topic sqs_test \
                --output_kafka_cluster_connection_string localhost:9094 \
            none \
            listen \
                --args queue_url=https://sqs.us-east-1.amazonaws.com/123456789012/my-queue batch_size=10 batch_interval=10
        ```

        ## Using geniusrise to invoke via YAML file
        ```yaml
        version: "1"
        spouts:
            my_sqs_spout:
                name: "SQS"
                method: "listen"
                args:
                    queue_url: "https://sqs.us-east-1.amazonaws.com/123456789012/my-queue"
                    batch_size: 10
                    batch_interval: 10
                output:
                    type: "streaming"
                    args:
                        output_topic: "sqs_test"
                        kafka_servers: "localhost:9094"
        ```
        """
        super().__init__(output, state)
        self.top_level_arguments = kwargs
        self.sqs = boto3.client("sqs")

    def listen(self, queue_url: str, batch_size: int = 10, batch_interval: int = 10):
        """
        📖 Start listening for new messages in the SQS queue.

        Args:
            queue_url (str): The URL of the SQS queue to listen to.
            batch_size (int): The maximum number of messages to receive in each batch. Defaults to 10.
            batch_interval (int): The time in seconds to wait for a new message if the queue is empty. Defaults to 10.

        Raises:
            Exception: If unable to connect to the SQS service.
        """
        self.queue_url = queue_url
        while True:
            try:
                # Receive message from SQS queue
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    AttributeNames=["All"],
                    MaxNumberOfMessages=batch_size,
                    MessageAttributeNames=["All"],
                    VisibilityTimeout=batch_interval,
                    WaitTimeSeconds=batch_interval,
                )

                if "Messages" in response:
                    for message in response["Messages"]:
                        receipt_handle = message["ReceiptHandle"]

                        # Enrich the data with metadata about the message ID
                        enriched_data = {
                            "data": message,
                            "message_id": message["MessageId"],
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

                        # Delete received message from queue
                        self.sqs.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
                else:
                    self.log.debug("No messages available in the queue.")
            except Exception as e:
                self.log.error(f"Error processing SQS message: {e}")

                # Update the state using the state
                current_state = self.state.get_state(self.id) or {
                    "success_count": 0,
                    "failure_count": 0,
                }
                current_state["failure_count"] += 1
                self.state.set_state(self.id, current_state)
