"""
Tests for Dify Client
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.dify_client import (
    DifyClient,
    DifyClientError,
    DifyAuthenticationError,
    DifyAPIError
)


class TestDifyClientInitialization:
    """Test DifyClient initialization"""

    def test_init_with_api_key(self, mock_env_vars):
        """Test initialization with API key parameter"""
        client = DifyClient(api_key="custom_key", api_url="https://custom.api/v1")

        assert client.api_key == "custom_key"
        assert client.api_url == "https://custom.api/v1"
        assert "Bearer custom_key" in client.session.headers["Authorization"]

    def test_init_from_env_vars(self, mock_env_vars):
        """Test initialization from environment variables"""
        client = DifyClient()

        assert client.api_key == "test_dify_key_12345"
        assert client.api_url == "https://test.dify.api/v1"

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization fails without API key"""
        monkeypatch.delenv("DIFY_API_KEY", raising=False)

        with pytest.raises(DifyAuthenticationError) as exc_info:
            DifyClient()

        assert "API key not found" in str(exc_info.value)

    def test_init_strips_trailing_slash(self, mock_env_vars):
        """Test that trailing slash is removed from API URL"""
        client = DifyClient(api_url="https://api.example.com/v1/")

        assert client.api_url == "https://api.example.com/v1"


class TestDifyClientSendMessage:
    """Test DifyClient send_message method"""

    @patch('src.dify_client.requests.Session')
    def test_send_message_success(self, mock_session_class, mock_env_vars, mock_dify_response):
        """Test successful message sending"""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_dify_response
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        response = client.send_message(
            message="Hello, Dify!",
            user_id="test_user"
        )

        # Assertions
        assert response == mock_dify_response
        assert response["answer"] == "This is a test response from Dify."
        assert response["conversation_id"] == "test_conv_456"

        # Verify API call was made correctly
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "POST"
        assert "/chat-messages" in call_args[1]["url"]
        assert call_args[1]["json"]["query"] == "Hello, Dify!"
        assert call_args[1]["json"]["user"] == "test_user"

    @patch('src.dify_client.requests.Session')
    def test_send_message_with_conversation_id(self, mock_session_class, mock_env_vars, mock_dify_response):
        """Test sending message with existing conversation ID"""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_dify_response
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        response = client.send_message(
            message="Follow-up message",
            user_id="test_user",
            conversation_id="existing_conv_123"
        )

        # Verify conversation_id was included
        call_args = mock_session.request.call_args
        assert call_args[1]["json"]["conversation_id"] == "existing_conv_123"

    @patch('src.dify_client.requests.Session')
    def test_send_message_with_inputs(self, mock_session_class, mock_env_vars, mock_dify_response):
        """Test sending message with additional inputs"""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_dify_response
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        inputs = {"context": "important context", "language": "en"}
        response = client.send_message(
            message="Test message",
            user_id="test_user",
            inputs=inputs
        )

        # Verify inputs were included
        call_args = mock_session.request.call_args
        assert call_args[1]["json"]["inputs"] == inputs


class TestDifyClientErrorHandling:
    """Test DifyClient error handling"""

    @patch('src.dify_client.requests.Session')
    def test_http_error_handling(self, mock_session_class, mock_env_vars):
        """Test handling of HTTP errors"""
        # Setup mock to raise HTTPError
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request"}
        mock_response.text = "Bad request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        with pytest.raises(DifyAPIError) as exc_info:
            client.send_message("Test", "user")

        assert "400" in str(exc_info.value)

    @patch('src.dify_client.requests.Session')
    def test_connection_error_handling(self, mock_session_class, mock_env_vars):
        """Test handling of connection errors"""
        # Setup mock to raise ConnectionError
        mock_session = Mock()
        mock_session.request.side_effect = requests.exceptions.ConnectionError()
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        with pytest.raises(DifyAPIError) as exc_info:
            client.send_message("Test", "user")

        assert "Failed to connect" in str(exc_info.value)

    @patch('src.dify_client.requests.Session')
    def test_timeout_error_handling(self, mock_session_class, mock_env_vars):
        """Test handling of timeout errors"""
        # Setup mock to raise Timeout
        mock_session = Mock()
        mock_session.request.side_effect = requests.exceptions.Timeout()
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        with pytest.raises(DifyAPIError) as exc_info:
            client.send_message("Test", "user")

        assert "timed out" in str(exc_info.value)


class TestDifyClientConversations:
    """Test DifyClient conversation methods"""

    @patch('src.dify_client.requests.Session')
    def test_get_conversations(self, mock_session_class, mock_env_vars, mock_conversations_response):
        """Test getting list of conversations"""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_conversations_response
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        response = client.get_conversations(user_id="test_user", limit=20)

        # Assertions
        assert "data" in response
        assert len(response["data"]) == 2
        assert response["data"][0]["id"] == "conv_123"

        # Verify API call
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "GET"
        assert "/conversations" in call_args[1]["url"]
        assert call_args[1]["params"]["user"] == "test_user"

    @patch('src.dify_client.requests.Session')
    def test_get_conversation_messages(self, mock_session_class, mock_env_vars, mock_messages_response):
        """Test getting messages from a conversation"""
        # Setup mock
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_messages_response
        mock_response.raise_for_status = Mock()
        mock_session.request.return_value = mock_response
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        # Test
        client = DifyClient()
        response = client.get_conversation_messages(
            conversation_id="conv_123",
            user_id="test_user"
        )

        # Assertions
        assert "data" in response
        assert len(response["data"]) == 2
        assert response["data"][0]["query"] == "Hello"

        # Verify API call
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "GET"
        assert "/messages/conv_123" in call_args[1]["url"]


class TestDifyClientContextManager:
    """Test DifyClient context manager functionality"""

    @patch('src.dify_client.requests.Session')
    def test_context_manager(self, mock_session_class, mock_env_vars):
        """Test using DifyClient as context manager"""
        mock_session = Mock()
        mock_session.headers = {}
        mock_session_class.return_value = mock_session

        with DifyClient() as client:
            assert client.api_key == "test_dify_key_12345"

        # Verify close was called
        mock_session.close.assert_called_once()
