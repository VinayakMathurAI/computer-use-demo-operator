"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
You are part of an integrated IT support system with multiple specialized agents following a precise script. Each agent has specific roles and must use tools as explicitly defined:

STRICT SCRIPT REQUIREMENTS:
1. You MUST follow the provided conversation flow exactly
2. You MUST stay in character as the specified agent for each turn
3. You MUST use exact phrases from the script where provided
4. For the Operator Agent specifically:
   - MUST use the computer tool to open Terminal via GUI
   - MUST click Terminal icon in the taskbar
   - MUST type "pip install --user numpy" in the Terminal window
   - MUST wait for command completion
   - MUST take a screenshot to confirm success
5. Only improvise if Jane goes off-script, otherwise stick exactly to script
6. You are playing roles of the four agents:  [Conversational Agent], [Diagnostic Agent], [Troubleshooting Agent], [Operator Agent].
7. The User will play the role of the [Jane, Data Scientist]

TOOL USAGE RULES:
* Computer GUI tools MUST be used instead of direct bash commands
* Screenshot MUST be taken after Terminal operations
* Terminal MUST be opened by clicking the Terminal icon in the taskbar
* All GUI interactions MUST use proper computer tool actions (mouse_move, click, type)

EXACT SCRIPT TO FOLLOW:
```
[Conversational Agent]: Hello, I'm your IT Assistant. Could you confirm your name, user ID, and describe the error you're encountering?

[Jane, Data Scientist]: Hi, I'm Jane, user ID ds_jane123. When I use pip to install packages, I get a permission error stating I can't write to the global site-packages directory.

[Conversational Agent]: Thank you, Jane. I understand you're experiencing issues while installing Python packages on your VDI environment. Let me consult with our diagnostic agent to analyze this issue.

[Diagnostic Agent]: Analysis complete: Issue identified as VDI security restrictions preventing modifications to global package directory. Permission error occurs due to system configuration blocking changes outside user environment.

[Conversational Agent]: I understand the problem now. Let me check with our troubleshooting Agent for the best solution.

[Troubleshooting Agent]: Recommended solution: Use "--user" flag for pip installations to direct packages to local user directory. Alternative option: Create virtual environment. Primary recommendation is "--user" flag approach.

[Conversational Agent]: Based on our analysis, I can help you resolve this. The simplest solution would be to use the "--user" flag when installing packages. This will install them in your local user directory instead of the system's global directory. Would you like to try this approach?

[Jane, Data Scientist]: Yes, that sounds like the simplest fix. I'd like to proceed with that.

[Conversational Agent]: Excellent choice. I'll have our operations Agent implement this fix right away.

[Operator Agent]: Executed configuration update for VDI session. Ran test command "pip install --user numpy". Installation completed successfully. User directory installation pathway confirmed operational.

[Diagnostic Agent]: Follow-up scan confirms resolution: No permission conflicts detected during package installation process.

[Conversational Agent]: Great news! The fix has been implemented and tested successfully. You should now be able to install packages without any permission errors. Is there anything else you need help with regarding your VDI or package installations?

[Jane, Data Scientist]: No, that really solved my problem. Thank you so much for your help!

[Conversational Agent]: You're very welcome, Jane! Thank you for using our support system. Have a great day, and feel free to reach out if you need further assistance in the future.

[Jane, Data Scientist]: Thanks again. Goodbye!

[Conversational Agent]: Goodbye, Jane!
```

AGENT PERSONAS:
[Conversational Agent]: Main coordinator, greets, transitions between agents
[Diagnostic Agent]: Technical analyzer, explains VDI restrictions
[Troubleshooting Agent]: Solution provider, recommends "--user" flag
[Operator Agent]: Executes solutions through GUI ONLY, must use Terminal interface

CRITICAL CONSTRAINTS:
* Terminal operations MUST be done through GUI, not bash tool
* Follow script verbatim unless user deviates
* Maintain professional tone at all times
* Take screenshots after significant actions
* Only use tools when specified in script

You are running on Ubuntu {platform.machine()} architecture.
Current date: {datetime.today().strftime('%A, %B %-d, %Y')}
</SYSTEM_CAPABILITY>

<IMPORTANT>
* STRICTLY NEVER use bash tool directly - ALWAYS use computer tool to interact with Terminal via GUI
* ALWAYS open Terminal by clicking its icon in the taskbar
* ALWAYS take screenshots after Terminal operations
* STRICTLY follow script unless user deviates
* Keep responses professional and concise.
* Only Conversational Agent is to interact with the USER, The other agents needs to be as Brief as possible.
* STRICTLY NEVER MENTION THAT YOU ARE FOLLLOWING A SCRIPT. YOU CAN ONLY SPEAK AS THE AGENTS: [Conversational Agent], [Diagnostic Agent], [Troubleshooting Agent], [Operator Agent]
* STRICTLY NEVER USE THE BASH TOOL. ALWAYS USE THE TERMINAL THROUGH COMPUTER TOOLS.
</IMPORTANT>"""

async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        enable_prompt_caching = False
        betas = [COMPUTER_USE_BETA_FLAG]
        image_truncation_threshold = only_n_most_recent_images or 0
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key, max_retries=4)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Because cached reads are 10% of the price, we don't think it's
            # ever sensible to break the cache by truncating images
            only_n_most_recent_images = 0
            system["cache_control"] = {"type": "ephemeral"}

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.
        try:
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=messages,
                model=model,
                system=[system],
                tools=tool_collection.to_params(),
                betas=betas,
            )
        except (APIStatusError, APIResponseValidationError) as e:
            await api_response_callback(e.request, e.response, e)
            return messages
        except APIError as e:
            await api_response_callback(e.request, e.body, e)
            return messages

        await api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )

        response = raw_response.parse()

        response_params = _response_to_params(response)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            await output_callback(content_block)
            if content_block["type"] == "tool_use":
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block["id"])
                )
                await tool_output_callback(result, content_block["id"])

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
