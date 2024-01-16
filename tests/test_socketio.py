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
from unittest import mock

import pytest
from geniusrise import InMemoryState, State, StreamingOutput

from geniusrise_audio.socketio import SocketIo


@pytest.fixture
def mock_output():
    return mock.MagicMock(spec=StreamingOutput)


@pytest.fixture
def mock_state():
    return mock.MagicMock(spec=State)


@pytest.fixture
def mock_socketio():
    with mock.patch("socketio.Client") as mock_sio:
        yield mock_sio


def test_socketio_init(mock_output, mock_state, mock_socketio):
    socketio_instance = SocketIo(mock_output, mock_state, arg1="value1")

    assert socketio_instance.top_level_arguments == {"arg1": "value1"}
    mock_socketio.assert_called_once()


def test_socketio_message_handler_str(mock_output, mock_state, mock_socketio):
    socketio_instance = SocketIo(mock_output, mock_state)
    mock_output.save = mock.MagicMock()
    message = '{"key": "value"}'

    socketio_instance._message_handler(message)

    mock_output.save.assert_called_once_with({"key": "value"})


def test_socketio_message_handler_dict(mock_output, mock_state, mock_socketio):
    socketio_instance = SocketIo(mock_output, mock_state)
    mock_output.save = mock.MagicMock()
    message = {"key": "value"}

    socketio_instance._message_handler(message)

    mock_output.save.assert_called_once_with(message)


def test_socketio_message_handler_invalid(mock_output, mock_state, mock_socketio):
    socketio_instance = SocketIo(mock_output, mock_state)
    message = [1, 2, 3]

    with pytest.raises(ValueError):
        socketio_instance._message_handler(message)


def test_socketio_listen(mock_output, mock_state, mock_socketio):
    socketio_instance = SocketIo(mock_output, mock_state)
    mock_sio_instance = mock_socketio.return_value
    url = "http://localhost:3000"
    namespace = "/chat"
    event = "message"

    socketio_instance.listen(url=url, namespace=namespace)

    mock_sio_instance.connect.assert_called_once_with(url, namespaces=[namespace])
    mock_sio_instance.on.assert_called_once_with(event, namespace=namespace)


def test_socketio_listen_exception(mock_output, mock_socketio):
    ims = InMemoryState()
    socketio_instance = SocketIo(mock_output, ims)
    mock_sio_instance = mock_socketio.return_value
    mock_sio_instance.connect.side_effect = Exception("Failed to connect")
    url = "http://localhost:3000"
    namespace = "/chat"

    socketio_instance.listen(url=url, namespace=namespace)

    print(json.dumps(socketio_instance.state.get_state(socketio_instance.id), indent=4))

    assert socketio_instance.state.get_state(socketio_instance.id)["failure_count"] == 10023
