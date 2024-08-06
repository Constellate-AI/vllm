import json
import re
from typing import Optional, Union, List, Dict

import partial_json_parser
from partial_json_parser import Allow
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, \
    AutoTokenizer

from vllm.entrypoints.openai.protocol import ExtractedToolCallInformation, \
    ToolCall, FunctionCall, DeltaMessage, DeltaToolCall, DeltaFunctionCall, \
    InitialDeltaToolCall
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
from vllm.logger import init_logger

logger = init_logger(__name__)


class Hermes2ProToolParser(ToolParser):
    tool_call_start_token: str = '<tool_call>'
    tool_call_end_token: str = '</tool_call>'

    # regex to match between <tool_call> and </tool_call> OR between <tool_call>
    # and EOS (happens sometimes :))
    tool_call_regex = re.compile(
        r'<tool_call>(.*?)</tool_call>|<tool_call>(.*)', re.DOTALL)
    scratch_pad_regex = re.compile(r'<scratch_pad>(.*?)</scratch_pad>',
                                   re.DOTALL)

    @staticmethod
    def extract_tool_calls(model_output: str) -> ExtractedToolCallInformation:

        # sanity check; avoid unnecessary processing
        if Hermes2ProToolParser.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        else:

            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = (
                    Hermes2ProToolParser.tool_call_regex.findall(model_output))

                # load the JSON, and then use it to build the Function and
                # Tool Call
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type='function',
                        function=FunctionCall(
                            name=function_call['name'],
                            # function call args are JSON but as a string
                            arguments=json.dumps(function_call['arguments'])))
                    for function_call in raw_function_calls
                ]

                content = model_output[:model_output.find(
                    Hermes2ProToolParser.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None)

            except Exception as e:
                logger.error("Error in extracting tool call from response %s",
                             e)
                return ExtractedToolCallInformation(tools_called=False,
                                                    tool_calls=[],
                                                    content=model_output)

    def __init__(self,
                 tokenizer: Optional[Union[PreTrainedTokenizer,
                                           PreTrainedTokenizerFast,
                                           AutoTokenizer]] = None):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: List[Dict] = []
        self.current_tool_id: int = -1
        self.current_tool_name_sent = False
        self.current_tool_initial_sent: bool = False
        self.streamed_args_for_tool: List[str] = [
        ]  # map what has been streamed for each tool so far to a list

        if not self.model_tokenizer:
            raise ValueError(
                'The model tokenizer must be passed to the ToolParser '
                'constructor during construction.')
        self.tool_call_start_token_id: int = self.model_tokenizer.vocab[
            '<tool_call>']
        self.tool_call_end_token_id: int = self.model_tokenizer.vocab[
            '</tool_call>']
        if not self.tool_call_start_token_id or not self.tool_call_end_token_id:
            raise RuntimeError(
                'Hermes 2 Pro Tool parser could not locate tool call start/end '
                'tokens in the tokenizer!')

    def extract_tool_calls_streaming(
            self, previous_text: str, current_text: str, delta_text: str,
            previous_token_ids: List[int], current_token_ids: List[int],
            delta_token_ids: List[int]) -> Union[DeltaMessage, None]:

        logger.debug(f'delta_text: {delta_text}')
        logger.debug(f'delta_token_ids: {delta_token_ids}')
        # check to see if we should be streaming a tool call - is there a
        if self.tool_call_start_token_id not in current_token_ids:
            logger.debug('No tool call tokens found!')
            return DeltaMessage(content=delta_text)

        else:
            try:

                # figure out where we are in the parsing by counting tool call
                # start & end tags
                prev_tool_start_count = previous_token_ids.count(
                    self.tool_call_start_token_id)
                prev_tool_end_count = previous_token_ids.count(
                    self.tool_call_end_token_id)
                cur_tool_start_count = current_token_ids.count(
                    self.tool_call_start_token_id)
                cur_tool_end_count = current_token_ids.count(
                    self.tool_call_end_token_id)

                # a cheap case - we're generating text, NOT tool calls.
                if (cur_tool_start_count == cur_tool_end_count
                        and prev_tool_end_count == cur_tool_end_count):
                    logger.debug(
                        'Generating text content! skipping tool parsing.')
                    return DeltaMessage(content=delta_text)

                # most of the time, we're going in here - we need to do partial
                # JSON parsing and build stuff.
                else:
                    # flags for partial JSON parting. exported constants from
                    # "Allow" are handled via BIT MASK
                    flags = Allow.ALL if self.current_tool_name_sent \
                        else Allow.ALL & ~Allow.STR

                    # if a new tool call is being started. unusual since
                    # normally the first "cheap case" will be hit.
                    if (cur_tool_start_count > cur_tool_end_count
                            and cur_tool_start_count > prev_tool_start_count):
                        if len(delta_token_ids) > 1:
                            tool_call_portion = current_text.split(
                                self.tool_call_start_token)[-1]
                            text_portion = None
                        else:
                            tool_call_portion = None
                            text_portion = None
                            delta = None

                        # set cursors and state appropriately
                        self.current_tool_id += 1
                        self.current_tool_name_sent = False
                        self.current_tool_initial_sent = False
                        self.streamed_args_for_tool.append('')
                        logger.debug(
                            f'Starting on a new tool {self.current_tool_id}')

                    # if an existing tool call is being updated - the most
                    # common case!
                    elif (cur_tool_start_count > cur_tool_end_count
                          and cur_tool_start_count == prev_tool_start_count):
                        tool_call_portion = current_text.split(
                            self.tool_call_start_token)[-1]
                        text_portion = None

                    # if the current tool call is being closed
                    elif (cur_tool_start_count == cur_tool_end_count
                          and cur_tool_end_count > prev_tool_end_count):
                        logger.debug('Closing the current tool call!')
                        diff = self.prev_tool_call_arr[
                            self.current_tool_id].get('arguments')
                        if diff:
                            diff = json.dumps(diff).replace(
                                self.streamed_args_for_tool[
                                    self.current_tool_id], '')
                            logger.debug(
                                f'Finishing tool and found diff that had not '
                                f'been streamed yet: {diff}')
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(index=self.current_tool_id,
                                              function=DeltaFunctionCall(
                                                  arguments=diff).model_dump(
                                                      exclude_none=True))
                            ])

                    else:
                        logger.error(
                            'INVARIANT - invalid state trying to parse tool '
                            'calls (wtf?)')
                        delta = None
                        return delta

                    logger.debug(f'Tool call portion: {tool_call_portion}')
                    current_tool_call = partial_json_parser.loads(
                        tool_call_portion,
                        flags) if tool_call_portion else None
                    logger.debug(f'Parsed tool call {current_tool_call}')

                    # make sure to send the initial message first if we haven't
                    # already - with the tool ID
                    if not self.current_tool_initial_sent:
                        logger.debug('Sending InitialDeltaToolCall')
                        self.current_tool_initial_sent = True
                        return DeltaMessage(tool_calls=[
                            InitialDeltaToolCall(
                                index=self.current_tool_id).model_dump(
                                    exclude_none=True)
                        ])

                    # after that, make sure we send the function name before
                    # any arguments
                    elif not self.current_tool_name_sent:
                        function_name: Union[
                            str, None] = current_tool_call.get('name')
                        if function_name:
                            logger.debug(
                                f'Sending DeltaToolCall with function name '
                                f'{function_name}!')
                            self.current_tool_name_sent = True
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(index=self.current_tool_id,
                                              function=DeltaFunctionCall(
                                                  name=function_name).
                                              model_dump(exclude_none=True))
                            ])
                        else:
                            return None
                    else:
                        # if there is no tool calls
                        if tool_call_portion is None:
                            # if there's text but not tool calls, send that -
                            # otherwise None to skip chunk
                            delta = DeltaMessage(
                                content=delta_text
                            ) if text_portion is not None else None
                        # now, the nitty-gritty of tool calls
                        else:
                            # now we have the portion to parse as tool call.
                            if text_portion is not None:
                                logger.debug(f'Also, will send text portion '
                                             f'{text_portion}')

                            logger.debug(
                                f'Trying to parse current tool call with ID '
                                f'{self.current_tool_id}')
                            if len(self.prev_tool_call_arr
                                   ) <= self.current_tool_id:
                                self.prev_tool_call_arr.append({})
                                logger.debug(
                                    'Pushed dummy value into tool call arr')
                            # main logic for tool parsing here
                            prev_arguments = self.prev_tool_call_arr[
                                self.current_tool_id].get('arguments')
                            cur_arguments = current_tool_call.get(
                                'arguments'
                            )  # arguments, if any, in current dict

                            logger.debug(
                                f'Diffing old arguments {prev_arguments} '
                                f'against new ones {cur_arguments}')
                            if not cur_arguments and not prev_arguments:
                                logger.debug(
                                    f'Skipping text {delta_text} - no arguments'
                                )
                                delta = None
                            elif not cur_arguments and prev_arguments:
                                logger.error(
                                    'INVARIANT - impossible to have arguments '
                                    'reset mid-call')
                                delta = None
                            elif cur_arguments and not prev_arguments:
                                cur_arguments_json = json.dumps(cur_arguments)
                                logger.debug(f'Finding {delta_text} in '
                                             f'{cur_arguments_json}')
                                arguments_delta = cur_arguments_json[:
                                                                     cur_arguments_json
                                                                     .index(
                                                                         delta_text
                                                                     ) +
                                                                     len(delta_text
                                                                         )]
                                logger.debug(
                                    f'First tokens in arguments received:'
                                    f' {arguments_delta}')
                                delta = DeltaMessage(tool_calls=[
                                    DeltaToolCall(index=self.current_tool_id,
                                                  function=DeltaFunctionCall(
                                                      arguments=arguments_delta
                                                  ).model_dump(
                                                      exclude_none=True))
                                ])
                                self.streamed_args_for_tool[
                                    self.current_tool_id] += arguments_delta

                            elif cur_arguments and prev_arguments:
                                cur_args_json = json.dumps(cur_arguments)
                                prev_args_json = json.dumps(prev_arguments)
                                logger.debug(
                                    f"Searching for diff between "
                                    f"\n{cur_args_json}\n{prev_args_json}")
                                argument_diff = extract_intermediate_diff(
                                    cur_args_json, prev_args_json)
                                logger.debug(
                                    f'Got argument diff: {argument_diff}')
                                delta = DeltaMessage(tool_calls=[
                                    DeltaToolCall(index=self.current_tool_id,
                                                  function=DeltaFunctionCall(
                                                      arguments=argument_diff).
                                                  model_dump(
                                                      exclude_none=True))
                                ])
                                self.streamed_args_for_tool[
                                    self.current_tool_id] += argument_diff
                            else:
                                delta = None

                            # handle saving the state for the current tool into
                            # the "prev" list for use in diffing for
                            # the next iteration
                            if self.current_tool_id == len(
                                    self.prev_tool_call_arr) - 1:
                                self.prev_tool_call_arr[
                                    self.current_tool_id] = current_tool_call
                            else:
                                self.prev_tool_call_arr.append(
                                    current_tool_call)

                            # TODO REPLACE ME WITH TOOL CALL
                            # delta = DeltaMessage(content=delta_text)
                        return delta

            except Exception as e:
                logger.error(
                    f'Error trying to handle streaming tool call: {e}')
                logger.debug(
                    'Skipping chunk as a result of tool streaming extraction '
                    'error')
                return None  # do not stream a delta. skip this token ID.
