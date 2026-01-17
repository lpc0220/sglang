# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Conversation chat templates for DeepSeek models.

This module provides conversation template definitions, data structures, and utilities
for managing chat templates for DeepSeek models in SGLang.

Key components:
- Conversation class: Defines the structure and behavior of chat templates
- SeparatorStyle enum: Different conversation formatting styles
- Template registry: Functions to register and retrieve templates by name or model path
- Built-in templates: Pre-defined templates for DeepSeek models
"""

# Adapted from
# https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
import dataclasses
import json
import os
import re
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

from typing_extensions import Literal

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.utils import ImageData


class SeparatorStyle(IntEnum):
    """Separator styles for DeepSeek models."""

    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    DEEPSEEK_CHAT = auto()
    DeepSeekVL2 = auto()
    PADDLE_OCR = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_TWO
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # The string that represents an image token in the prompt
    image_token: str = "<image>"
    video_token: str = "<video>"
    audio_token: str = "<audio>"

    image_data: Optional[List[ImageData]] = None
    video_data: Optional[List[str]] = None
    modalities: Optional[List[str]] = None
    stop_token_ids: Optional[int] = None

    audio_data: Optional[List[str]] = None
    image_token_at_prefix: bool = False

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DeepSeekVL2:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.PADDLE_OCR:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": "
                    if role == self.roles[0]:
                        if self.image_token in message:
                            ret += message.replace(
                                self.image_token + "\n", self.image_token
                            )
                        else:
                            ret += message
                        ret += "\n"
                    else:
                        ret += message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def append_image(self, image: str, detail: Literal["auto", "low", "high"]):
        """Append a new image."""
        self.image_data.append(ImageData(url=image, detail=detail))

    def append_video(self, video: str):
        """Append a new video."""
        self.video_data.append(video)

    def append_audio(self, audio: str):
        """Append a new audio."""
        self.audio_data.append(audio)

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        if self.system_message == "":
            ret = []
        else:
            ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            image_token=self.image_token,
            video_token=self.video_token,
            audio_token=self.audio_token,
            image_token_at_prefix=self.image_token_at_prefix,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
chat_templates: Dict[str, Conversation] = {}
matching_function_registry: List[Callable] = []


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in chat_templates
        ), f"{template.name} has been registered."

    chat_templates[template.name] = template


def register_conv_template_matching_function(func):
    matching_function_registry.append(func)


def get_conv_template_by_model_path(model_path):
    for matching_func in matching_function_registry:
        conv_name = matching_func(model_path)
        if conv_name is not None:
            return conv_name
    return None


def chat_template_exists(template_name: str) -> bool:
    return template_name in chat_templates


def generate_embedding_convs(
    texts: List[str], images: List[str], videos: List[str], template_name: str
) -> List[Conversation]:
    conv_template = chat_templates[template_name].copy()
    convs = []
    for text, image, video in zip(texts, images, videos):
        conv = Conversation(
            name=conv_template.name,
            system_template=conv_template.system_template,
            system_message=conv_template.system_message,
            roles=conv_template.roles,
            messages=list(conv_template.messages),  # prevent in-place modification
            offset=conv_template.offset,
            sep_style=SeparatorStyle(conv_template.sep_style),
            sep=conv_template.sep,
            sep2=conv_template.sep2,
            stop_str=conv_template.stop_str,
            image_data=[],
            video_data=[],
            audio_data=[],
            modalities=[],
            image_token=conv_template.image_token,
            video_token=conv_template.video_token,
            audio_token=conv_template.audio_token,
            image_token_at_prefix=conv_template.image_token_at_prefix,
        )
        real_content = ""

        if image is not None:
            image_token = conv.image_token + "\n"
            real_content += image_token
        if video is not None:
            real_content += conv.video_token
        if text is not None:
            real_content += text
        conv.append_message(conv.roles[0], real_content)
        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        convs.append(conv)

    return convs


# Models in which system adds modality tokens at prompt start automatically
# when media inputs exceed modality tokens in prompt (e.g. 3 images but 2 <image> tokens)
_MODELS_REQUIRING_MODALITY_SUPPLEMENT = {"deepseek-vl2"}


# adapted from https://github.com/vllm-project/vllm/blob/5124f5bf51b83e6f344c1bc6652e8c4d81313b34/vllm/entrypoints/chat_utils.py#L856
def _get_full_multimodal_text_prompt(
    modality_token: str, modality_count: int, text_prompt: str
) -> str:
    """Combine multimodal prompts for a multimodal language model."""

    # For any existing placeholder in the text prompt, we leave it as is
    left: int = modality_count - text_prompt.count(modality_token)
    if left < 0:
        raise ValueError(
            f"Found more '{modality_token}' placeholders in input prompt than "
            "actual multimodal data items."
        )

    # NOTE: For now we always add missing modality_token at the front of
    # the prompt. This may change to be customizable in the future.
    return "\n".join([modality_token] * left + [text_prompt])


def generate_chat_conv(
    request: ChatCompletionRequest, template_name: str
) -> Conversation:
    conv = chat_templates[template_name].copy()
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        image_data=[],
        video_data=[],
        audio_data=[],
        modalities=[],
        image_token=conv.image_token,
        audio_token=conv.audio_token,
        video_token=conv.video_token,
        image_token_at_prefix=conv.image_token_at_prefix,
    )

    if isinstance(request.messages, str):
        raise ValueError("The messages should be a list of dict.")
    for message in request.messages:
        msg_role = message.role
        if msg_role == "system":
            if isinstance(message.content, str):
                conv.system_message = message.content
            elif isinstance(message.content, list):
                if (
                    len(message.content) != 1
                    or getattr(message.content[0], "type", None) != "text"
                ):
                    raise ValueError("The system message should be a single text.")
                else:
                    conv.system_message = getattr(message.content[0], "text", "")
        elif msg_role == "user":
            # Handle the various types of Chat Request content types here.
            if isinstance(message.content, str):
                conv.append_message(conv.roles[0], message.content)
            else:
                real_content = ""
                # calculate number of image_url
                num_image_url = 0
                for content in message.content:
                    if content.type == "image_url":
                        num_image_url += 1
                        conv.modalities.append(content.modalities)
                image_token = conv.image_token + "\n"
                add_token_as_needed: bool = (
                    conv.name in _MODELS_REQUIRING_MODALITY_SUPPLEMENT
                )
                if add_token_as_needed:
                    image_token = ""

                audio_token = conv.audio_token
                video_token = conv.video_token
                for content in message.content:
                    if content.type == "text":
                        if num_image_url > 16:
                            real_content += "\n"  # for video
                        real_content += content.text
                    elif content.type == "image_url":
                        if conv.image_token_at_prefix:
                            real_content = image_token + real_content
                        else:
                            real_content += image_token
                        conv.append_image(
                            content.image_url.url, content.image_url.detail
                        )
                    elif content.type == "video_url":
                        real_content += video_token
                        conv.append_video(content.video_url.url)
                    elif content.type == "audio_url":
                        real_content += audio_token
                        conv.append_audio(content.audio_url.url)
                if add_token_as_needed:
                    real_content = _get_full_multimodal_text_prompt(
                        conv.image_token, num_image_url, real_content
                    )
                conv.append_message(conv.roles[0], real_content)
        elif msg_role == "assistant":
            parsed_content = ""
            if isinstance(message.content, str):
                parsed_content = message.content
            elif isinstance(message.content, list):
                if (
                    len(message.content) != 1
                    or getattr(message.content[0], "type", None) != "text"
                ):
                    raise ValueError(
                        "The assistant's response should be a single text."
                    )
                else:
                    parsed_content = getattr(message.content[0], "text", "")
            conv.append_message(conv.roles[1], parsed_content)
        else:
            raise ValueError(f"Unknown role: {msg_role}")

    # Add a blank message for the assistant.
    conv.append_message(conv.roles[1], None)
    return conv


# ==============================================================================
# DeepSeek Model Templates
# ==============================================================================

# DeepSeek OCR template
register_conv_template(
    Conversation(
        name="deepseek-ocr",
        system_message="",
        system_template="",
        roles=("", ""),
        sep="",
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        stop_str=["<｜end▁of▁sentence｜>"],
        image_token="<image>",
        image_token_at_prefix=True,
    )
)

# DeepSeek VL2 template
register_conv_template(
    Conversation(
        name="deepseek-vl2",
        system_template="{system_message}",
        system_message="",
        roles=("<|User|>", "<|Assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DeepSeekVL2,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str=["User:", "<｜end▁of▁sentence｜>"],
    )
)

# DeepSeek Janus Pro template
# Reference: https://github.com/deepseek-ai/Janus?tab=readme-ov-file#janus-pro
register_conv_template(
    Conversation(
        name="janus-pro",
        system_message="You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language",
        system_template="{system_message}.",
        roles=("User", "Assistant"),
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        stop_str=["<|User|>", "<｜end▁of▁sentence｜>"],
        image_token="<image_placeholder>",
    )
)

# Model type to template mapping for DeepSeek models
MODEL_TYPE_TO_TEMPLATE = {
    "deepseek_vl_v2": "deepseek-vl2",
    "multi_modality": "janus-pro",
    "deepseek-ocr": "deepseek-ocr",
}


def get_model_type(model_path: str) -> Optional[str]:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_type")
    except (IOError, json.JSONDecodeError):
        return None


@register_conv_template_matching_function
def match_deepseek_janus_pro(model_path: str):
    if re.search(r"janus", model_path, re.IGNORECASE):
        return "janus-pro"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_deepseek_vl(model_path: str):
    if re.search(r"deepseek.*vl2", model_path, re.IGNORECASE):
        return "deepseek-vl2"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)


@register_conv_template_matching_function
def match_deepseek_ocr(model_path: str):
    if "deepseek-ocr" in model_path.lower():
        return "deepseek-ocr"
    model_type = get_model_type(model_path)
    return MODEL_TYPE_TO_TEMPLATE.get(model_type)
