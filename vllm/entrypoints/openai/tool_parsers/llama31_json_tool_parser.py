from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              InitialDeltaToolCall, ToolCall)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from typing import List, Dict, Union, Sequence

logger = init_logger(__name__)


class Llama31JsonToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = 1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

        # llama 3.1 doesn't use a start token it just response with JSON
        # opening with the tool call name
        self.tool_call_start_token: str = "<|python_tag|>"
        # closing the parameters dict
        self.tool_call_end_str: str = "}}"

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the "
                             "ToolParser constructor.")

    def extract_tool_calls(
            self,
            model_output: str
    ) -> ExtractedToolCallInformation:

        if not model_output.startswith(self.tool_call_start_token):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )

        #tool_text = model_output.replace(self.tool_call_start_token, '')
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output
        )


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