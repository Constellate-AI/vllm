import json
from typing import Dict, List, Optional, Union

import partial_json_parser
from partial_json_parser import Allow
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.entrypoints.openai.protocol import (DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall,
                                              InitialDeltaToolCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger

logger = init_logger(__name__)


# IMPORTANT
# this parser is built around assumptions about how parallel tool calling is
# handled, which Llama 3.1 is NOT designed to do natively. It will do them
# consistently IF AND ONLY IF you use the prompt template:
# examples/tool_chat_template_llama_3_1.jinja
# otherwise you should not count on parallel tool calls working.
class Llama31ToolParser(ToolParser):
    # we are using the "JSON-based tool calling" functionality of the model
    # more on this here:
    # https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
    tool_call_start_text: str = '{"name": '

    def __init__(self, tokenizer: Optional[Union[PreTrainedTokenizer,
                                                 PreTrainedTokenizerFast,
                                                 AutoTokenizer]]):
        super().__init__(tokenizer)

        self.prev_tool_call_arr: List[dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = []

    @staticmethod
    def extract_tool_calls(model_output: str) -> ExtractedToolCallInformation:

        logger.info(
            'Checking to see if should extract tool calls from output %s',
            model_output)

        if not ('"name"' in model_output and '"parameters"' in model_output):
            logger.info('Llama 3.1 did not generate a tool call')
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)
        else:
            generated_function_calls = model_output.split('; ')
            function_calls: List[Dict] = []

            # parse them one at a time so if some are wrong we get others right
            for call in generated_function_calls:
                try:
                    function_calls.append(json.loads(call))

                except json.decoder.JSONDecodeError as e:
                    logger.error('Error trying to decode function call: %s', e)

            # parse them into tool calls
            tool_calls: List[ToolCall] = []
            for function_call in function_calls:

                try:
                    function = FunctionCall(name=function_call['name'],
                                            arguments=json.dumps(
                                                function_call['parameters']))
                    tool_calls.append(ToolCall(function=function))
                except KeyError as e:
                    logger.error('malformed function call generated: %s', e)
                except Exception as e:
                    logger.error('unable to parse function call: %s', e)

                logger.error('Generated (erroneous) call: %s', model_output)

            return ExtractedToolCallInformation(tools_called=True,
                                                tool_calls=tool_calls,
                                                content='')

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: List[int],
        current_token_ids: List[int],
        delta_token_ids: List[int],
    ) -> Union[DeltaMessage, None]:

        print('current text: %s', current_text)
        try:
            logger.info('delta_text: %s', delta_text)

            # check if we should start streaming a tool call, or text
            if (self.tool_call_start_text not in current_text
                    and current_text not in self.tool_call_start_text):
                logger.info('LLama 3 is not generating a tool call')
                return DeltaMessage(content=delta_text)

            # configure bitmask flags for partial JSON parsing
            partial_json_parsing_flags = Allow.ALL \
                if self.current_tool_name_sent \
                else Allow.ALL & ~Allow.STR

            # TODO review this
            # if a new tool call is being started - they will be separated by
            # a semicolon and space ???
            generated_calls = current_text.split('; ')

            # if a new call is generated, we need to reset our values
            if len(generated_calls) > self.current_tool_id + 1:
                logger.info('Starting a new tool call!')
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.current_tool_initial_sent = False
                self.streamed_args_for_tool.append(
                    '')  # add an entry for this tool
                self.prev_tool_call_arr.append({})

                logger.info('Set current tool id to %s', self.current_tool_id)

                # if nonzero index,
                # we need to make sure we closed the old one too
                if self.current_tool_id > 0:
                    # get the old tool call
                    old_tool_call = self.prev_tool_call_arr[
                        self.current_tool_id - 1]
                    logger.info('Checking old tool call')
                    logger.info('Old tool call: %s', old_tool_call)

                    # get its arguments
                    old_arguments = old_tool_call.get('arguments')
                    logger.info('Got old arguments from tool call')
                    logger.info('old arguments: %s', old_arguments)

                    # dump its parameters, and compare it to the arguments that
                    # were streamed so far
                    diff = json.dumps(old_arguments).replace(
                        self.streamed_args_for_tool[self.current_tool_id - 1],
                        '')
                    logger.info('Got diff: %s', diff)
                    if diff:
                        return DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_id - 1,
                                          function=DeltaFunctionCall(
                                              arguments=diff).model_dump(
                                                  exclude_none=True))
                        ])

            # grab only the text for the call we're working with
            working_text_for_function_call = generated_calls[
                self.current_tool_id]
            logger.info('Working on tool call %s: %s', self.current_tool_id,
                        working_text_for_function_call)

            # use the partial JSON parser to load the current call
            current_function_call = partial_json_parser.loads(
                working_text_for_function_call, partial_json_parsing_flags)
            logger.info('parsed function call %s', current_function_call)

            # the first chunk should always contain the tool ID
            # - if we haven't sent that yet, then do that
            if not self.current_tool_initial_sent:
                logger.info('Sending InitialDeltaToolCall')
                self.current_tool_initial_sent = True
                return DeltaMessage(tool_calls=[
                    InitialDeltaToolCall(
                        index=self.current_tool_id).model_dump(
                            exclude_none=True)
                ])

            # the next chunk always contains the tool name -
            # if we haven't; do that
            # yet either, do that tool
            elif not self.current_tool_name_sent:

                # remember if it hasn't been sent yet, flags don't allow for
                # partial fields - so unless we have the full name this
                # will be None
                function_name: Optional[str] = current_function_call.get(
                    'name')
                if function_name:
                    logger.info('Sending DeltaToolCall with function name %s',
                                function_name)
                    self.current_tool_name_sent = True
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_id,
                                      function=DeltaFunctionCall(
                                          name=function_name).model_dump(
                                              exclude_none=True))
                    ])
                else:
                    # because we ALWAYS send the name first per OAI - don't let
                    # anything else stream until we're able to send the name
                    return None

            # now, we know we're ready to send argument chunks
            current_function_arguments = current_function_call.get(
                'parameters')
            prev_function_arguments = self.prev_tool_call_arr[
                self.current_tool_id].get('arguments')

            logger.info('diffing old arguments %s against current ones %s',
                        prev_function_arguments, current_function_arguments)

            if not current_function_arguments and not prev_function_arguments:
                logger.info('Skipping text %s - no arguments are ready',
                            delta_text)

            cur_args_json = json.dumps(current_function_arguments)
            # if there are no "previous" arguments
            if not prev_function_arguments:
                logger.info('Finding %s in %s', delta_text, cur_args_json)
                arguments_delta = cur_args_json[:cur_args_json.index(delta_text
                                                                     ) +
                                                len(delta_text)]
                logger.info('first Tokens in arguments received: %s',
                            arguments_delta)

            # if there are previous arguments and we're diffing
            else:
                prev_args_json = json.dumps(prev_function_arguments)
                logger.info('Searching for diff between\n%s\n%s',
                            cur_args_json, prev_args_json)

                arguments_delta = extract_intermediate_diff(
                    cur_args_json, prev_args_json)
                logger.info('got argument diff: %s', arguments_delta)

            # based on the diff, create the delta
            delta = DeltaMessage(tool_calls=[
                DeltaToolCall(index=self.current_tool_id,
                              function=DeltaFunctionCall(
                                  arguments=arguments_delta).model_dump(
                                      exclude_none=True))
            ])
            self.streamed_args_for_tool[
                self.current_tool_id] += arguments_delta
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = {
                    'name': current_function_call.get('name'),
                    'arguments': current_function_call.get('parameters')
                }
            else:
                self.prev_tool_call_arr.append({
                    'name':
                    current_function_call.get('name'),
                    'arguments':
                    current_function_call.get('parameters')
                })

            return delta

        except Exception as e:
            logger.error('Error trying to handle streaming tool call: %s', e)
            logger.info(
                'Skipping chunk as a result of tool streaming extraction '
                'error')
            return None  # do not stream a delta. skip this token ID.
