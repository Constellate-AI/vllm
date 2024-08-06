import json, re
from typing import Optional, List, Union, Dict
import partial_json_parser
from partial_json_parser import Allow
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer
)
from vllm.entrypoints.openai.protocol import (
    ExtractedToolCallInformation, ToolCall, FunctionCall, DeltaMessage,
    DeltaFunctionCall, DeltaToolCall, InitialDeltaToolCall
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger

logger = init_logger(__name__)

class Llama31ToolParser(ToolParser):

    # we are using the "JSON-based tool calling" functionality of the model
    # more on this here:
    # https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
    tool_call_start_text: str = '{"name": '

    def __init__(self, tokenizer: Optional[Union[
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        AutoTokenizer
    ]]):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: List[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

    @staticmethod
    def extract_tool_calls(model_output: str) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: List[int],
        current_token_ids: List[int],
        delta_token_ids: List[int],
    ) -> Union[DeltaMessage, None]:
        pass
