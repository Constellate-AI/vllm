# OpenAI Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) API.

You can start the server using Python, or using [Docker](deploying_with_docker.rst):
```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

To call the server, you can use the official OpenAI Python client library, or any other HTTP client.
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

## API Reference
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference) for more information on the API. We support all parameters except:
- Chat: `tools`, and `tool_choice`.
- Completions: `suffix`.

vLLM also provides experimental support for OpenAI Vision API compatible inference. See more details in [Using VLMs](../models/vlm.rst).

## Extra Parameters
vLLM supports a set of parameters that are not part of the OpenAI API.
In order to use them, you can pass them as extra parameters in the OpenAI client.
Or directly merge them into the JSON payload if you are using HTTP call directly.

```python
completion = client.chat.completions.create(
  model="NousResearch/Meta-Llama-3-8B-Instruct",
  messages=[
    {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
  ],
  extra_body={
    "guided_choice": ["positive", "negative"]
  }
)
```

### Extra Parameters for Chat API
The following [sampling parameters (click through to see documentation)](../dev/sampling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-chat-completion-sampling-params
:end-before: end-chat-completion-sampling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-chat-completion-extra-params
:end-before: end-chat-completion-extra-params
```

### Extra Parameters for Completions API
The following [sampling parameters (click through to see documentation)](../dev/sampling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-completion-sampling-params
:end-before: end-completion-sampling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-completion-extra-params
:end-before: end-completion-extra-params
```

## Chat Template

In order for the language model to support chat protocol, vLLM requires the model to include
a chat template in its tokenizer configuration. The chat template is a Jinja2 template that
specifies how are roles, messages, and other chat-specific tokens are encoded in the input.

An example chat template for `NousResearch/Meta-Llama-3-8B-Instruct` can be found [here](https://github.com/meta-llama/llama3?tab=readme-ov-file#instruction-tuned-models)

Some models do not provide a chat template even though they are instruction/chat fine-tuned. For those model,
you can manually specify their chat template in the `--chat-template` parameter with the file path to the chat
template, or the template in string form. Without a chat template, the server will not be able to process chat
and all chat requests will error.

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM community provides a set of chat templates for popular models. You can find them in the examples
directory [here](https://github.com/vllm-project/vllm/tree/main/examples/)

## Command line arguments for the server

```{argparse}
:module: vllm.entrypoints.openai.cli_args
:func: create_parser_for_docs
:prog: vllm serve
```
## Tool Calling in the Chat Completion API
### Named Function Calling
vLLM supports only named function calling in the chat completion API by default. It does so using Outlines, so this is 
enabled by default, and will work with any supported model. You are guaranteed a validly-parsable function call - not a 
high-quality one. 

To use a named function, you need to define the functions in the `tools` parameter of the chat completion request, and 
specify the `name` of one of the tools in the `tool_choice` parameter of the chat completion request. 

It is the callers responsibility to prompt the model with the tool information, vLLM will not automatically manipulate the prompt.

vLLM will use guided decoding to ensure the response matches the tool parameter object defined by the JSON schema in the `tools` parameter.


### Automatic Function Calling
_This feature is in **beta**. It has limited model support, is not guaranteed to be stable, and does not have 
well-defined failure modes._ As such, it must be explicitly enabled when desired.

To enable this feature, you must set the following flags:
* `--enable-auto-tool-choice` -- **mandatory** Auto tool choice. tells vLLM that you want to enable the model to generate its own tool calls when it 
deems appropriate. 
* `--chat-template` -- **optional** for auto tool choice. the path to the chat template which handles `tool`-role messages and `assistant`-role messages 
that contain previously generated tool calls.This argument can be set to `tool_use` if your model has a tool use chat 
template configured in the `tokenizer_config.json`. In this case, it will be used per the `transformers` specification. More on this [here](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)
from HuggingFace; and you can find an example of this in a `tokenizer_config.json` [here](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)
* `--tool-parser` -- select the tool parser to use - currently either `hermes` or `mistral`. 

If your favorite tool-calling model is not supported, please feel free to contribute a parser & tool use chat template! 

#### Hermes Models
Supported models in this series:
* `NousResearch/Hermes-2-Pro-Llama-3-8B`
* `NousResearch/Hermes-2-Theta-Llama-3-70B`
* `NousResearch/Hermes-2-Pro-Llama-3-70B`
* `NousResearch/Hermes-2-Theta-Llama-3-8B`
* `NousResearch/Hermes-2-Pro-Mistral-7B`

_Note that the Hermes 2 **Theta** models are known to have degraded tool call quality & capabilities due to the merge 
step in their creation_. It is recommended to use the Hermes 2 **Pro** models. 

Recommended flags: `--tool-call-parser hermes --chat-template examples/tool_chat_template_hermes.jinja`

#### Mistral Models
Supported models:
* `mistralai/Mistral-7B-Instruct-v0.3`
* Possibly mistral-large and mixtral? These have not been tested at the time of this writing.

Known issues:
1. Mistral 7B struggles to generate parallel tool calls correctly. 
2. Mistral's `tokenizer_config.json` chat template requires tool call IDs that are exactly 9 digits, which is 
much shorter than what vLLM generates. 

To address this, the following additional chat templates are provided:

* `examples/tool_chat_template_mistral.jinja` - this is the "official" Mistral chat template, but tweaked so that
it works with vLLM's tool call IDs (provided `tool_call_id` fields are truncated to the last 9 digits)
* `examples/tool_chat_template_mistral_parallel.jinja` - this is a "better" version that adds a tool-use system prompt
when tools are provided, that results in much better reliability when working with parallel tool calling.

**Please note** that the model's default chat template in `tokenizer_config.json` will not work with vLLM, as it expects
tool_call_id fields to be exactly 9 digits, which is shorter than vLLM's format. You **must** do one of the following 
to get tool calling to work with mistral:
1. use one of the 2 provided tool chat templates
2. provide your own tool chat template that corrects for this
3. in your client code, ignore the vLLM-generated `tool_call_id`, and manually generate and pass in your own 9-digit
`tool_call_id`s for `assistant`-role messages containing tool calls, and `tool`-role messages containing tool call 
results.

Recommended flags: `--tool-call-parser mistral --chat-template examples/tool_chat_template_mistral.jinja`
