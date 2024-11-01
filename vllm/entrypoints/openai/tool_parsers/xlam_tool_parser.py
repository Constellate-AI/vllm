import json
import re
from typing import Dict, List, Sequence, Union
import partial_json_parser
from partial_json_parser.core.options import Allow
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall,
                                              DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              ToolCall)

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager
)
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("xlam")
class SalesforceXlamToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if isinstance(self.model_tokenizer, MistralTokenizer):
            logger.error(
                "Detected Mistral tokenizer when using XLAM tool parser")
            self.model_tokenizer = self.model_tokenizer.tokenizer

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: List[str] = []

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser"
                "constructor during construction")

        # TODO set up start/end tokens if necesary; does XLAM geneate text?


    def extract_tool_calls(
            self,
            model_output: str,
            request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:

        # TODO
        return ExtractedToolCallInformation(
            tools_caooled=False,
            tool_calls=[],
            content=model_output
        )

    def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request: ChatCompletionRequest
    ) -> Union[DeltaMessage, None]:

        # TODO
        return None