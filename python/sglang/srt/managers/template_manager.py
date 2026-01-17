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
"""
DeepSeek-only build: Simplified template management.

This module provides a minimal interface for template management.
The full conversation/chat template system has been removed in favor of
using HuggingFace tokenizer chat templates directly.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def detect_jinja_template_content_format(template: str) -> str:
    """
    Detect the content format expected by the Jinja chat template.

    Returns:
        'openai' if the template expects OpenAI-style message content format
        'string' if the template expects simple string content
    """
    if template is None:
        return "string"

    # Check for patterns that indicate OpenAI format (handling content as list of objects)
    openai_indicators = [
        r"content\s*\|\s*selectattr",  # Jinja filter for handling content types
        r"\.type\s*==\s*['\"]text['\"]",  # Type checking for content parts
        r"content\.text",  # Accessing text from content object
        r"for\s+\w+\s+in\s+content",  # Iterating over content parts
    ]

    for pattern in openai_indicators:
        if re.search(pattern, template):
            return "openai"

    return "string"


class TemplateManager:
    """
    DeepSeek-only build: Simplified template manager.

    Uses HuggingFace tokenizer chat templates directly instead of
    custom conversation templates.
    """

    def __init__(self):
        self._chat_template_name: Optional[str] = None
        self._completion_template_name: Optional[str] = None
        self._jinja_template_content_format: Optional[str] = "openai"
        self._force_reasoning: bool = False

    @property
    def chat_template_name(self) -> Optional[str]:
        """Get the current chat template name."""
        return self._chat_template_name

    @property
    def completion_template_name(self) -> Optional[str]:
        """Get the current completion template name."""
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> Optional[str]:
        """Get the detected template content format ('string' or 'openai' or None)."""
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool:
        """
        Check if the current chat template enforces reasoning/thinking.

        Returns:
            True if the template contains reasoning patterns like <think> tags
        """
        return self._force_reasoning

    def _detect_reasoning_pattern(self, template: str) -> bool:
        """
        Detect if the chat template contains reasoning/thinking patterns.
        """
        if template is None:
            return False

        # DeepSeek-R1 reasoning pattern
        force_reasoning_pattern = r"<\|im_start\|>assistant\\n<think>\\n"
        has_reasoning = re.search(force_reasoning_pattern, template) is not None

        if has_reasoning:
            logger.info("Detected the force reasoning pattern in chat template.")

        return has_reasoning

    def load_chat_template(
        self, tokenizer_manager, chat_template_arg: Optional[str], model_path: str
    ) -> None:
        """
        Load a chat template from various sources.

        In DeepSeek-only build, we primarily use HuggingFace tokenizer chat templates.
        """
        if chat_template_arg:
            # Load explicit template from file
            if chat_template_arg.endswith(".jinja"):
                self._load_jinja_template(tokenizer_manager, chat_template_arg)
            else:
                logger.warning(
                    f"Chat template '{chat_template_arg}' not supported in DeepSeek-only build. "
                    "Using HuggingFace tokenizer chat template instead."
                )
                self._load_hf_template(tokenizer_manager)
        else:
            # Use HuggingFace template
            self._load_hf_template(tokenizer_manager)

        # Detect reasoning pattern from chat template
        if tokenizer_manager.tokenizer:
            self._force_reasoning = self._detect_reasoning_pattern(
                tokenizer_manager.tokenizer.chat_template
            )

    def _load_hf_template(self, tokenizer_manager) -> None:
        """Load template from HuggingFace tokenizer."""
        hf_template = self._resolve_hf_chat_template(tokenizer_manager)
        if hf_template:
            self._jinja_template_content_format = detect_jinja_template_content_format(
                hf_template
            )
            logger.info(
                f"Using HuggingFace chat template with detected content format: {self._jinja_template_content_format}"
            )
        else:
            self._jinja_template_content_format = "string"
            logger.info(
                "No chat template found, defaulting to 'string' content format"
            )

    def _load_jinja_template(self, tokenizer_manager, template_path: str) -> None:
        """Load a Jinja template file."""
        with open(template_path, "r") as f:
            chat_template = "".join(f.readlines()).strip("\n")
        tokenizer_manager.tokenizer.chat_template = chat_template.replace("\\n", "\n")
        self._chat_template_name = None
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            f"Loaded Jinja chat template with content format: {self._jinja_template_content_format}"
        )

    def guess_chat_template_from_model_path(self, model_path: str) -> None:
        """
        Infer chat template name from model path.

        DeepSeek-only build: No pre-defined templates, uses HF tokenizer templates.
        """
        # DeepSeek models use HuggingFace tokenizer chat templates
        pass

    def load_completion_template(self, completion_template_arg: str) -> None:
        """
        Load completion template for code completion.

        DeepSeek-only build: Code completion templates not supported.
        """
        logger.warning(
            f"Completion template '{completion_template_arg}' not supported in DeepSeek-only build."
        )

    def initialize_templates(
        self,
        tokenizer_manager,
        model_path: str,
        chat_template: Optional[str] = None,
        completion_template: Optional[str] = None,
    ) -> None:
        """
        Initialize all templates based on provided configuration.
        """
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

        if completion_template:
            self.load_completion_template(completion_template)

    def _resolve_hf_chat_template(self, tokenizer_manager) -> Optional[str]:
        """
        Resolve HuggingFace chat template.

        Returns the chat template string if found, None otherwise.
        """
        try:
            if processor := tokenizer_manager.processor:
                if hasattr(processor, "chat_template") and processor.chat_template:
                    return processor.chat_template
            if tokenizer := tokenizer_manager.tokenizer:
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                    return tokenizer.chat_template
        except Exception as e:
            logger.debug(f"Error getting chat template: {e}")

        logger.debug("No HuggingFace chat template found")
        return None
