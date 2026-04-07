import hashlib
import logging
import re
from copy import deepcopy
from typing import Any, Dict, Optional

from mem0.configs.prompts import (
    FACT_RETRIEVAL_PROMPT,
    USER_MEMORY_EXTRACTION_PROMPT,
    AGENT_MEMORY_EXTRACTION_PROMPT,
    SESSION_MEMORY_EXTRACTION_PROMPT,
)
from mem0.exceptions import ValidationError as Mem0ValidationError

logger = logging.getLogger(__name__)


def get_fact_retrieval_messages(message, is_agent_memory=False, is_multi_speaker_memory=False):
    """Get fact retrieval messages based on the memory type.

    Args:
        message: The message content to extract facts from
        is_agent_memory: If True, use agent memory extraction prompt, else use user memory extraction prompt
        is_multi_speaker_memory: If True, use session memory extraction prompt for multi-speaker conversations

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    if is_agent_memory:
        return AGENT_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
    elif is_multi_speaker_memory:
        return SESSION_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
    else:
        return USER_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"


def get_fact_retrieval_messages_legacy(message):
    """Legacy function for backward compatibility."""
    return FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"


def is_multi_speaker_format(message):
    """Check if messages are in multi-speaker format."""
    if not isinstance(message, dict) or not message:
        return False
    # Check if first message has 'speaker' field with 'id' and 'name'
    return (
        isinstance(message, dict)
        and "speaker" in message
        and isinstance(message["speaker"], dict)
        and "id" in message["speaker"]
        and "name" in message["speaker"]
        and "content" in message
    )


def parse_messages(messages):
    """Parse messages supporting both OpenAI format and multi-speaker format.

    OpenAI format: [{"role": "user", "content": "xxx"}, ...]
    Multi-speaker format: [{"speaker": {"id": "1", "name": "a"}, "content": "xxx"}, ...]
    """
    response = ""

    # Process each message individually based on its format
    for msg in messages:
        # Check if this specific message has speaker format
        if is_multi_speaker_format(msg):
            speaker_name = msg["speaker"]["name"]
            speaker_id = msg["speaker"]["id"]
            content = msg["content"]
            role = msg.get("role", "")
            if role:
                response += f"{role}#{speaker_name}#{speaker_id}: {content}\n"
            else:
                response += f"{speaker_name} # {speaker_id}: {content}\n"
        elif msg.get("role") and msg.get("content"):
            # Traditional role format
            role = msg.get("role")
            content = msg.get("content", "")
            response += f"{role}: {content}\n"

    return response.strip()


def format_entities(entities):
    if not entities:
        return ""

    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relationship']} -- {entity['destination']}"
        formatted_lines.append(simplified)

    return "\n".join(formatted_lines)


def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    match_res=match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", match_res, flags=re.DOTALL).strip()



def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text  # assume it's raw JSON
    return json_str


def get_image_description(image_obj, llm, vision_details):
    """
    Get the description of the image
    """

    if isinstance(image_obj, str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A user is providing an image. Provide a high level description of the image and do not include any additional text.",
                    },
                    {"type": "image_url", "image_url": {"url": image_obj, "detail": vision_details}},
                ],
            },
        ]
    else:
        messages = [image_obj]

    response = llm.generate_response(messages=messages)
    return response


def parse_vision_messages(messages, llm=None, vision_details="auto"):
    """
    Parse the vision messages from the messages
    """
    returned_messages = []
    for msg in messages:
        if msg["role"] == "system":
            returned_messages.append(msg)
            continue

        # Handle message content
        if isinstance(msg["content"], list):
            # Multiple image URLs in content
            description = get_image_description(msg, llm, vision_details)
            returned_messages.append({"role": msg["role"], "content": description})
        elif isinstance(msg["content"], dict) and msg["content"].get("type") == "image_url":
            # Single image content
            image_url = msg["content"]["image_url"]["url"]
            try:
                description = get_image_description(image_url, llm, vision_details)
                returned_messages.append({"role": msg["role"], "content": description})
            except Exception:
                raise Exception(f"Error while downloading {image_url}.")
        else:
            # Regular text content
            returned_messages.append(msg)

    return returned_messages


def process_telemetry_filters(filters):
    """
    Process the telemetry filters
    """
    if filters is None:
        return {}

    encoded_ids = {}
    if "user_id" in filters:
        encoded_ids["user_id"] = hashlib.md5(filters["user_id"].encode()).hexdigest()
    if "agent_id" in filters:
        encoded_ids["agent_id"] = hashlib.md5(filters["agent_id"].encode()).hexdigest()
    if "run_id" in filters:
        encoded_ids["run_id"] = hashlib.md5(filters["run_id"].encode()).hexdigest()

    return list(filters.keys()), encoded_ids


def sanitize_relationship_for_cypher(relationship) -> str:
    """Sanitize relationship text for Cypher queries by replacing problematic characters."""
    char_map = {
        "...": "_ellipsis_",
        "…": "_ellipsis_",
        "。": "_period_",
        "，": "_comma_",
        "；": "_semicolon_",
        "：": "_colon_",
        "！": "_exclamation_",
        "？": "_question_",
        "（": "_lparen_",
        "）": "_rparen_",
        "【": "_lbracket_",
        "】": "_rbracket_",
        "《": "_langle_",
        "》": "_rangle_",
        "'": "_apostrophe_",
        '"': "_quote_",
        "\\": "_backslash_",
        "/": "_slash_",
        "|": "_pipe_",
        "&": "_ampersand_",
        "=": "_equals_",
        "+": "_plus_",
        "*": "_asterisk_",
        "^": "_caret_",
        "%": "_percent_",
        "$": "_dollar_",
        "#": "_hash_",
        "@": "_at_",
        "!": "_bang_",
        "?": "_question_",
        "(": "_lparen_",
        ")": "_rparen_",
        "[": "_lbracket_",
        "]": "_rbracket_",
        "{": "_lbrace_",
        "}": "_rbrace_",
        "<": "_langle_",
        ">": "_rangle_",
    }

    # Apply replacements and clean up
    sanitized = relationship
    for old, new in char_map.items():
        sanitized = sanitized.replace(old, new)

    return re.sub(r"_+", "_", sanitized).strip("_")


def safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to JSON serialization for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using JSON serialization: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump(mode="json")
            except Exception:
                clone_dict = {k: v for k, v in config.__dict__.items()}
        elif hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            clone_dict = asdict(config)
        else:
            clone_dict = {k: v for k, v in config.__dict__.items()}

        sensitive_tokens = ("auth", "credential", "password", "token", "secret", "key", "connection_class")
        for field_name in list(clone_dict.keys()):
            if any(token in field_name.lower() for token in sensitive_tokens):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception as reconstruction_error:
            logger.warning(
                f"Failed to reconstruct config: {reconstruction_error}. "
                f"Telemetry may be affected."
            )
            raise


def build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.
    """
    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of user_id, agent_id, or run_id must be provided.",
            error_code="VALIDATION_001",
            details={"user_id": user_id, "agent_id": agent_id, "run_id": run_id},
            suggestion="Provide at least one session identifier for proper scoping."
        )

    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters
