import re

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
import json
from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              InitialDeltaToolCall, ToolCall)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from typing import List, Dict, Union, Sequence, Optional

logger = init_logger(__name__)


class Llama31JsonToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = 1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

        # llama 3.1 sometimes uses <|python_tag|> and sometimes uses plain JSON
        self.tool_call_start_str = '<|python_tag|>'

        self.tool_call_regex = re.compile(
            r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}',
            re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the "
                             "ToolParser constructor.")

    def extract_tool_calls(self,
                           model_output: str) -> ExtractedToolCallInformation:

        call_text: Optional[str] = None

        # check if it starts with <|python_tag|>
        if model_output.startswith(self.tool_call_start_str):
            call_text = model_output.replace(self.tool_call_start_str, "")

        # OR looks like it might be calling tools without it
        elif "name" in model_output and "parameters" in model_output:
            call_text = model_output

        if not call_text:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:

            tool_call_text = self.tool_call_regex.findall(call_text)
            tool_call = json.loads(tool_call_text[0])

            # llama 3.1 does NOT support parallel tool calls
            if not isinstance(tool_call, dict):
                raise Exception(
                    "Generated tool call could not be parsed to JSON")
            name = tool_call.get("name", None)
            parameters = tool_call.get("parameters", None)
            if not name:
                raise Exception("Generated tool call does not have a name!")
            if not parameters:
                raise Exception(
                    "Generated tool call does not have 'parameters'!")

            tool_calls = [
                ToolCall(type="function",
                         function=FunctionCall(
                             name=name, arguments=json.dumps(parameters)))
            ]
            content = call_text.replace(tool_call_text[0], "")
            return ExtractedToolCallInformation(tools_called=True,
                                                tool_calls=tool_calls,
                                                content=content)

        except Exception as e:
            logger.error("Error extracting tool call from response: %s", e)
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    # noop
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        pass
