"""
Pytest configuration and shared fixtures
"""

import pytest
from unittest.mock import Mock, MagicMock
import os


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing"""
    monkeypatch.setenv("DIFY_API_KEY", "test_dify_key_12345")
    monkeypatch.setenv("DIFY_API_URL", "https://test.dify.api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_12345")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_dify_response():
    """Mock successful Dify API response"""
    return {
        "event": "message",
        "message_id": "test_msg_123",
        "conversation_id": "test_conv_456",
        "mode": "chat",
        "answer": "This is a test response from Dify.",
        "metadata": {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        },
        "created_at": 1234567890
    }


@pytest.fixture
def mock_conversations_response():
    """Mock Dify conversations list response"""
    return {
        "data": [
            {
                "id": "conv_123",
                "name": "Test Conversation 1",
                "created_at": 1234567890
            },
            {
                "id": "conv_456",
                "name": "Test Conversation 2",
                "created_at": 1234567891
            }
        ],
        "has_more": False,
        "limit": 20
    }


@pytest.fixture
def mock_messages_response():
    """Mock Dify messages response"""
    return {
        "data": [
            {
                "id": "msg_123",
                "conversation_id": "conv_123",
                "query": "Hello",
                "answer": "Hi there!",
                "created_at": 1234567890
            },
            {
                "id": "msg_124",
                "conversation_id": "conv_123",
                "query": "How are you?",
                "answer": "I'm doing well, thank you!",
                "created_at": 1234567891
            }
        ],
        "has_more": False,
        "limit": 20
    }


@pytest.fixture
def mock_requests_session(mock_dify_response):
    """Mock requests.Session for API calls"""
    session = Mock()
    response = Mock()
    response.status_code = 200
    response.json.return_value = mock_dify_response
    response.raise_for_status = Mock()

    session.request.return_value = response
    session.headers = {}

    return session


@pytest.fixture
def mock_langchain_llm():
    """Mock LangChain LLM"""
    llm = Mock()
    llm.predict.return_value = "This is a mock LangChain response."
    return llm
