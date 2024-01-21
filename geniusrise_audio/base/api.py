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
import threading
from typing import Any, Dict, Optional

import cherrypy
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger

from .bulk import AudioBulk

# Define a global lock for sequential access control
sequential_lock = threading.Lock()


def sequential_tool():
    with sequential_lock:
        # Yield to signal that the request can proceed
        yield


# Register the custom tool
cherrypy.tools.sequential = cherrypy.Tool("before_handler", sequential_tool)


class AudioAPI(AudioBulk):
    """
    A class representing a Hugging Face API for generating text using a pre-trained language model.

    Attributes:
        model (Any): The pre-trained language model.
        processor (Any): The processor used to preprocess input text.
        model_name (str): The name of the pre-trained language model.
        model_revision (Optional[str]): The revision of the pre-trained language model.
        processor_name (str): The name of the processor used to preprocess input text.
        processor_revision (Optional[str]): The revision of the processor used to preprocess input text.
        model_class (str): The name of the class of the pre-trained language model.
        processor_class (str): The name of the class of the processor used to preprocess input text.
        use_cuda (bool): Whether to use a GPU for inference.
        quantization (int): The level of quantization to use for the pre-trained language model.
        precision (str): The precision to use for the pre-trained language model.
        device_map (str | Dict | None): The mapping of devices to use for inference.
        max_memory (Dict[int, str]): The maximum memory to use for inference.
        torchscript (bool): Whether to use a TorchScript-optimized version of the pre-trained language model.
        model_args (Any): Additional arguments to pass to the pre-trained language model.

    Methods:
        text(**kwargs: Any) -> Dict[str, Any]:
            Generates text based on the given prompt and decoding strategy.

        listen(model_name: str, model_class: str = "AutoModelForCausalLM", processor_class: str = "AutoProcessor", use_cuda: bool = False, precision: str = "float16", quantization: int = 0, device_map: str | Dict | None = "auto", max_memory={0: "24GB"}, torchscript: bool = True, endpoint: str = "*", port: int = 3000, cors_domain: str = "http://localhost:3000", username: Optional[str] = None, password: Optional[str] = None, **model_args: Any) -> None:
            Starts a CherryPy server to listen for requests to generate text.
    """

    model: Any
    processor: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes a new instance of the TextAPI class.

        Args:
            input (BatchInput): The input data to process.
            output (BatchOutput): The output data to process.
            state (State): The state of the API.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def validate_password(self, realm, username, password):
        """
        Validate the username and password against expected values.

        Args:
            realm (str): The authentication realm.
            username (str): The provided username.
            password (str): The provided password.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        return username == self.username and password == self.password

    def listen(
        self,
        model_name: str,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = False,
        compile: bool = True,
        endpoint: str = "*",
        port: int = 3000,
        cors_domain: str = "http://localhost:3000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **model_args: Any,
    ) -> None:
        """
        Starts a CherryPy server to listen for requests to generate text.

        Args:
            model_name (str): The name of the pre-trained language model.
            model_class (str, optional): The name of the class of the pre-trained language model. Defaults to "AutoModelForCausalLM".
            processor_class (str, optional): The name of the class of the processor used to preprocess input text. Defaults to "AutoProcessor".
            use_cuda (bool, optional): Whether to use a GPU for inference. Defaults to False.
            precision (str, optional): The precision to use for the pre-trained language model. Defaults to "float16".
            quantization (int, optional): The level of quantization to use for the pre-trained language model. Defaults to 0.
            device_map (str | Dict | None, optional): The mapping of devices to use for inference. Defaults to "auto".
            max_memory (Dict[int, str], optional): The maximum memory to use for inference. Defaults to {0: "24GB"}.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to True.
            compile (bool): Enable Torch JIT compilation.
            endpoint (str, optional): The endpoint to listen on. Defaults to "*".
            port (int, optional): The port to listen on. Defaults to 3000.
            cors_domain (str, optional): The domain to allow CORS requests from. Defaults to "http://localhost:3000".
            username (Optional[str], optional): The username to use for authentication. Defaults to None.
            password (Optional[str], optional): The password to use for authentication. Defaults to None.
            **model_args (Any): Additional arguments to pass to the pre-trained language model.
        """
        self.model_name = model_name
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.quantization = quantization
        self.precision = precision
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.compile = compile
        self.model_args = model_args
        self.username = username
        self.password = password

        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            processor_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            processor_name = model_name
        else:
            model_revision = None
            processor_revision = None
        processor_name = model_name
        self.model_name = model_name
        self.model_revision = model_revision
        self.processor_name = processor_name
        self.processor_revision = processor_revision

        self.model, self.processor = self.load_models(
            model_name=self.model_name,
            processor_name=self.processor_name,
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            model_class=self.model_class,
            processor_class=self.processor_class,
            use_cuda=self.use_cuda,
            precision=self.precision,
            quantization=self.quantization,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            compile=self.compile,
            **self.model_args,
        )

        def CORS():
            cherrypy.response.headers["Access-Control-Allow-Origin"] = cors_domain
            cherrypy.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            cherrypy.response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            cherrypy.response.headers["Access-Control-Allow-Credentials"] = "true"

            if cherrypy.request.method == "OPTIONS":
                cherrypy.response.status = 200
                return True

        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,
                "tools.CORS.on": True,
                "error_page.400": error_page,
                "error_page.401": error_page,
                "error_page.402": error_page,
                "error_page.403": error_page,
                "error_page.404": error_page,
                "error_page.405": error_page,
                "error_page.406": error_page,
                "error_page.408": error_page,
                "error_page.415": error_page,
                "error_page.429": error_page,
                "error_page.500": error_page,
                "error_page.501": error_page,
                "error_page.502": error_page,
                "error_page.503": error_page,
                "error_page.504": error_page,
                "error_page.default": error_page,
            }
        )

        if username and password:
            # Configure basic authentication
            conf = {
                "/": {
                    "tools.auth_basic.on": True,
                    "tools.auth_basic.realm": "geniusrise",
                    "tools.auth_basic.checkpassword": self.validate_password,
                    "tools.CORS.on": True,
                }
            }
        else:
            # Configuration without authentication
            conf = {"/": {"tools.CORS.on": True}}

        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", conf)
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()


def error_page(status, message, traceback, version):
    response = {
        "status": status,
        "message": message,
    }
    return json.dumps(response)
