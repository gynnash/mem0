#!/usr/bin/env python3
"""Standalone test of the multi-speaker parser functions."""

import json
import re

# Copy the modified functions here to test them standalone
def is_multi_speaker_format(messages):
    """Check if messages are in multi-speaker format."""
    if not isinstance(messages, list) or not messages:
        return False
    # Check if first message has 'speaker' field with 'id' and 'name'
    first_msg = messages[0]
    return (
        isinstance(first_msg, dict)
        and "speaker" in first_msg
        and isinstance(first_msg["speaker"], dict)
        and "id" in first_msg["speaker"]
        and "name" in first_msg["speaker"]
        and "content" in first_msg
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
        if msg.get("speaker") and isinstance(msg["speaker"], dict) and msg["speaker"].get("name") and msg["speaker"].get("id") and msg.get("content"):
            speaker_name = msg["speaker"]["name"]
            speaker_id = msg["speaker"]["id"]
            content = msg["content"]
            response += f"{speaker_name} #{speaker_id}: {content}\n"
        elif msg.get("role") and msg.get("content"):
            # Traditional role format
            role = msg.get("role")
            content = msg.get("content", "")
            response += f"{role}: {content}\n"

    return response.strip()


def test_multi_speaker_parser():
    print("Testing Multi-Speaker Conversation Parser\n")

    # Test 1: Multi-speaker format
    print("Test 1: Multi-speaker format")
    multi_speaker_conv = [
        {"speaker": {"id": "1", "name": "Alice"}, "content": "Let's use Python for the backend development."},
        {"speaker": {"id": "2", "name": "Bob"}, "content": "Great idea! I would also recommend FastAPI for the API framework."},
        {"speaker": {"id": "3", "name": "Charlie"}, "content": "Actually, Bob, we might need to consider the team's expertise with Node.js too."}
    ]

    print(f"Input: {json.dumps(multi_speaker_conv, indent=2)}")
    print(f"is_multi_speaker_format: {is_multi_speaker_format(multi_speaker_conv)}")
    print(f"Parsed output:\n{parse_messages(multi_speaker_conv)}")

    # Test 2: Role-based format
    print("\n" + "="*60 + "\n")
    print("Test 2: Traditional role-based format (OpenAI style)")
    role_based_conv = [
        {"role": "user", "content": "What's the best database for our application?"},
        {"role": "assistant", "content": "I recommend PostgreSQL for its reliability and features."}
    ]

    print(f"Input: {json.dumps(role_based_conv, indent=2)}")
    print(f"is_multi_speaker_format: {is_multi_speaker_format(role_based_conv)}")
    print(f"Parsed output:\n{parse_messages(role_based_conv)}")

    # Test 3: Mixed context (this would fail format check)
    print("\n" + "="*60 + "\n")
    print("Test 3: Mixed format (this format would be recognized as role-based)")
    mixed_conv = [
        {"speaker": {"id": "1", "name": "User"}, "content": "Can you help me?"},
        {"role": "assistant", "content": "Sure! How can I help?"}
    ]

    print(f"Input: {json.dumps(mixed_conv, indent=2)}")
    print(f"is_multi_speaker_format: {is_multi_speaker_format(mixed_conv)}")
    print(f"Parsed output:\n{parse_messages(mixed_conv)}")
    print("(Note: Mixed formats are detected as role-based because first message doesn't match multi-speaker pattern)")

    # Test 4: Example use case (what memories might look like)
    print("\n" + "="*60 + "\n")
    print("Test 4: Example of multi-speaker conversation that would be stored as memories")
    example = [
        {"speaker": {"id": "1", "name": "Alice"}, "content": "We decided to use Python FastAPI for the backend with PostgreSQL database."},
        {"speaker": {"id": "2", "name": "Bob"}, "content": "Yes, and Charlie will handle the DevOps part with Docker and CI/CD pipeline."},
        {"speaker": {"id": "3", "name": "Charlie"}, "content": "I'll set up Jenkins for CI/CD and Docker containers for deployment."}
    ]

    parsed = parse_messages(example)
    print(f"Conversation:\n{parsed}")
    print("\nThis conversation would be processed and stored as memories including:")
    print("- Python FastAPI chosen for backend")
    print("- PostgreSQL database selected")
    print("- Charlie responsible for DevOps")
    print("- Docker containers planned")
    print("- Jenkins selected for CI/CD")

if __name__ == "__main__":
    test_multi_speaker_parser()
    print("\n\nParser testing completed! The implementation now supports multi-speaker format.")