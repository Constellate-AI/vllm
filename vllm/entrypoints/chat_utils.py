import codecs
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Awaitable, Iterable, List, Optional, Union, cast, final

# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
# yapf: enable
# pydantic needs the TypedDict from typing_extensions
from pydantic import ConfigDict
from transformers import PreTrainedTokenizer
from typing_extensions import Required, TypedDict

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.utils import async_get_and_parse_image

logger = init_logger(__name__)


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[OpenAIChatCompletionContentPartParam,
                                       CustomChatCompletionContentPartParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: Optional[str]
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """
    tool_call_id: Optional[str]

    tool_calls: Optional[List[dict]]


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: Optional[str]  # optional IFF tool_calls is specified
    tool_call_id: Optional[str]
    name: Optional[str]
    tool_calls: Optional[List]


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    mm_futures: List[Awaitable[MultiModalDataDict]] = field(
        default_factory=list)


def load_chat_template(chat_template: Optional[str]) -> Optional[str]:
    if chat_template is None:
        return None
    try:
        with open(chat_template, "r") as f:
            resolved_chat_template = f.read()
    except OSError as e:
        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            msg = (f"The supplied chat template ({chat_template}) "
                   f"looks like a file path, but it failed to be "
                   f"opened. Reason: {e}")
            raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        resolved_chat_template = codecs.decode(chat_template, "unicode_escape")

    logger.info("Using supplied chat template:\n%s", resolved_chat_template)
    return resolved_chat_template


@lru_cache(maxsize=None)
def _image_token_str(model_config: ModelConfig,
                     tokenizer: PreTrainedTokenizer) -> Optional[str]:
    # TODO: Let user specify how to insert image tokens into prompt
    # (similar to chat template)
    model_type = model_config.hf_config.model_type
    if model_type == "phi3_v":
        # Workaround since this token is not defined in the tokenizer
        return "<|image_1|>"
    if model_type == "minicpmv":
        return "(<image>./</image>)"
    if model_type in ("blip-2", "chatglm", "fuyu", "paligemma"):
        # These models do not use image tokens in the prompt
        return None
    if model_type.startswith("llava"):
        return tokenizer.decode(model_config.hf_config.image_token_index)
    if model_type in ("chameleon", "internvl_chat"):
        return "<image>"
    raise TypeError(f"Unknown model type: {model_type}")


# TODO: Let user specify how to insert image tokens into prompt
# (similar to chat template)
def _get_full_image_text_prompt(image_token_str: str, text_prompt: str) -> str:
    """Combine image and text prompts for vision language model"""

    # NOTE: For now we assume all model architectures use the same
    # image + text prompt format. This may change in the future.
    return f"{image_token_str}\n{text_prompt}"


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    model_config: ModelConfig,
    tokenizer: PreTrainedTokenizer,
) -> ChatMessageParseResult:
    texts: List[str] = []
    mm_futures: List[Awaitable[MultiModalDataDict]] = []

    for part in parts:
        part_type = part["type"]
        if part_type == "text":
            text = cast(ChatCompletionContentPartTextParam, part)["text"]
            texts.append(text)
        elif part_type == "image_url":
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple 'image_url' input is currently not supported.")

            image_url = cast(ChatCompletionContentPartImageParam,
                             part)["image_url"]

            if image_url.get("detail", "auto") != "auto":
                logger.warning(
                    "'image_url.detail' is currently not supported and "
                    "will be ignored.")

            image_future = async_get_and_parse_image(image_url["url"])
            mm_futures.append(image_future)
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)

    if mm_futures:
        image_token_str = _image_token_str(model_config, tokenizer)
        if image_token_str is not None:
            if image_token_str in text_prompt:
                logger.warning(
                    "Detected image token string in the text prompt. "
                    "Skipping prompt formatting.")
            else:
                text_prompt = _get_full_image_text_prompt(
                    image_token_str=image_token_str,
                    text_prompt=text_prompt,
                )

    messages = [ConversationMessage(role=role, content=text_prompt)]

    return ChatMessageParseResult(messages=messages, mm_futures=mm_futures)


def parse_chat_message_content(
    message: ChatCompletionMessageParam,
    model_config: ModelConfig,
    tokenizer: PreTrainedTokenizer,
) -> ChatMessageParseResult:
    role = message.get('role')
    content = message.get("content")
    tool_call_id = message.get('tool_call_id')
    tool_calls = message.get('tool_calls')
    name = message.get('name', '')

    if content is None and tool_calls is None:
        return ChatMessageParseResult(messages=[], image_futures=[])

    if isinstance(content, str) or tool_calls:
        if role == 'tool':
            messages = [
                ConversationMessage(role=role,
                                    name=name,
                                    content=content,
                                    tool_call_id=tool_call_id)
            ]
        elif role == 'assistant':
            if tool_calls:
                messages = [
                    ConversationMessage(role=role,
                                        content=content,
                                        tool_calls=list(tool_calls))
                ]
            else:
                messages = [ConversationMessage(role=role, content=content)]
        else:
            messages = [ConversationMessage(role=role, content=content)]
        return ChatMessageParseResult(messages=messages, image_futures=[])
    elif isinstance(content, list):
        return _parse_chat_message_content_parts(role, content, model_config,
                                                 tokenizer)
