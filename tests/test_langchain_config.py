"""
Tests for LangChain Configuration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.langchain_config import (
    LangChainConfig,
    DifyLangChainBridge,
    create_bridge
)
from src.dify_client import DifyClient


class TestLangChainConfigInitialization:
    """Test LangChainConfig initialization"""

    @patch('src.langchain_config.ChatOpenAI')
    def test_init_with_openai(self, mock_chat_openai, mock_env_vars):
        """Test initialization with OpenAI provider"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )

        assert config.provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        mock_chat_openai.assert_called_once()

    @patch('src.langchain_config.ChatOpenAI')
    def test_init_with_gpt4(self, mock_chat_openai, mock_env_vars):
        """Test initialization with GPT-4"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=1000
        )

        assert config.model_name == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_init_without_api_key(self, monkeypatch):
        """Test initialization fails without OpenAI API key"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            LangChainConfig(provider="openai")

        assert "OPENAI_API_KEY not found" in str(exc_info.value)

    def test_init_with_unsupported_provider(self, mock_env_vars):
        """Test initialization with unsupported provider"""
        with pytest.raises(ValueError) as exc_info:
            LangChainConfig(provider="unsupported_provider")

        assert "Unsupported provider" in str(exc_info.value)


class TestLangChainConfigChains:
    """Test LangChain chain creation"""

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_simple_chain(self, mock_chat_openai, mock_env_vars):
        """Test creating a simple LLMChain"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(provider="openai")
        chain = config.create_simple_chain(
            template="What is {topic}?",
            input_variables=["topic"]
        )

        assert chain is not None
        assert hasattr(chain, 'run')

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_simple_chain_infers_variables(self, mock_chat_openai, mock_env_vars):
        """Test that simple chain infers input variables from template"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(provider="openai")
        chain = config.create_simple_chain(
            template="Translate {text} to {language}"
        )

        assert chain is not None

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_conversation_chain(self, mock_chat_openai, mock_env_vars):
        """Test creating a ConversationChain"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(provider="openai")
        chain = config.create_conversation_chain(
            memory_type="buffer"
        )

        assert chain is not None
        assert hasattr(chain, 'memory')

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_conversation_chain_with_window(self, mock_chat_openai, mock_env_vars):
        """Test creating ConversationChain with window memory"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(provider="openai")
        chain = config.create_conversation_chain(
            memory_type="window",
            window_size=3
        )

        assert chain is not None
        assert hasattr(chain, 'memory')

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_conversation_chain_invalid_memory(self, mock_chat_openai, mock_env_vars):
        """Test that invalid memory type raises error"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(provider="openai")

        with pytest.raises(ValueError) as exc_info:
            chain = config.create_conversation_chain(memory_type="invalid")

        assert "Unsupported memory type" in str(exc_info.value)

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_chat_prompt_chain(self, mock_chat_openai, mock_env_vars):
        """Test creating a chat-style prompt chain"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        config = LangChainConfig(provider="openai")
        chain = config.create_chat_prompt_chain(
            system_template="You are a helpful assistant.",
            human_template="Help me with {task}"
        )

        assert chain is not None


class TestDifyLangChainBridge:
    """Test DifyLangChainBridge functionality"""

    @patch('src.langchain_config.ChatOpenAI')
    @patch('src.dify_client.requests.Session')
    def test_bridge_initialization(self, mock_session, mock_chat_openai, mock_env_vars):
        """Test bridge initialization"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_session_instance = Mock()
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance

        dify_client = DifyClient()
        langchain_config = LangChainConfig(provider="openai")
        bridge = DifyLangChainBridge(dify_client, langchain_config)

        assert bridge.dify_client == dify_client
        assert bridge.langchain_config == langchain_config

    @patch('src.langchain_config.ChatOpenAI')
    @patch('src.dify_client.requests.Session')
    def test_process_with_langchain_then_dify(
        self,
        mock_session,
        mock_chat_openai,
        mock_env_vars,
        mock_dify_response
    ):
        """Test preprocessing with LangChain before sending to Dify"""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        mock_session_instance = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_dify_response
        mock_response.raise_for_status = Mock()
        mock_session_instance.request.return_value = mock_response
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance

        # Create bridge
        dify_client = DifyClient()
        langchain_config = LangChainConfig(provider="openai")
        bridge = DifyLangChainBridge(dify_client, langchain_config)

        # Create preprocessing chain
        preprocessing_chain = Mock()
        preprocessing_chain.run.return_value = "Enhanced message"

        # Test
        result = bridge.process_with_langchain_then_dify(
            message="Original message",
            user_id="test_user",
            preprocessing_chain=preprocessing_chain
        )

        # Assertions
        assert result["original_message"] == "Original message"
        assert result["processed_message"] == "Enhanced message"
        assert "dify_response" in result
        preprocessing_chain.run.assert_called_once()

    @patch('src.langchain_config.ChatOpenAI')
    @patch('src.dify_client.requests.Session')
    def test_process_with_dify_then_langchain(
        self,
        mock_session,
        mock_chat_openai,
        mock_env_vars,
        mock_dify_response
    ):
        """Test postprocessing Dify response with LangChain"""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        mock_session_instance = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_dify_response
        mock_response.raise_for_status = Mock()
        mock_session_instance.request.return_value = mock_response
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance

        # Create bridge
        dify_client = DifyClient()
        langchain_config = LangChainConfig(provider="openai")
        bridge = DifyLangChainBridge(dify_client, langchain_config)

        # Create postprocessing chain
        postprocessing_chain = Mock()
        postprocessing_chain.run.return_value = "Processed response"

        # Test
        result = bridge.process_with_dify_then_langchain(
            message="Test message",
            user_id="test_user",
            postprocessing_chain=postprocessing_chain
        )

        # Assertions
        assert result["original_message"] == "Test message"
        assert result["dify_answer"] == "This is a test response from Dify."
        assert result["processed_response"] == "Processed response"
        postprocessing_chain.run.assert_called_once()

    @patch('src.langchain_config.ChatOpenAI')
    @patch('src.dify_client.requests.Session')
    def test_hybrid_response(
        self,
        mock_session,
        mock_chat_openai,
        mock_env_vars,
        mock_dify_response
    ):
        """Test getting hybrid responses from both Dify and LangChain"""
        # Setup mocks
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        mock_session_instance = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_dify_response
        mock_response.raise_for_status = Mock()
        mock_session_instance.request.return_value = mock_response
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance

        # Create bridge
        dify_client = DifyClient()
        langchain_config = LangChainConfig(provider="openai")
        bridge = DifyLangChainBridge(dify_client, langchain_config)

        # Create LangChain chain
        langchain_chain = Mock()
        langchain_chain.run.return_value = "LangChain response"

        # Test
        result = bridge.hybrid_response(
            message="Test message",
            user_id="test_user",
            langchain_chain=langchain_chain,
            combine_strategy="concatenate"
        )

        # Assertions
        assert result["message"] == "Test message"
        assert result["dify_answer"] == "This is a test response from Dify."
        assert result["langchain_answer"] == "LangChain response"
        assert "Dify:" in result["combined_answer"]
        assert "LangChain:" in result["combined_answer"]


class TestHelperFunctions:
    """Test helper functions"""

    @patch('src.langchain_config.ChatOpenAI')
    def test_create_default_config(self, mock_chat_openai, mock_env_vars):
        """Test creating default configuration"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        from src.langchain_config import create_default_config

        config = create_default_config()

        assert config.provider == "openai"
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7

    @patch('src.langchain_config.ChatOpenAI')
    @patch('src.dify_client.requests.Session')
    def test_create_bridge_function(self, mock_session, mock_chat_openai, mock_env_vars):
        """Test create_bridge helper function"""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_session_instance = Mock()
        mock_session_instance.headers = {}
        mock_session.return_value = mock_session_instance

        bridge = create_bridge()

        assert isinstance(bridge, DifyLangChainBridge)
        assert bridge.dify_client is not None
        assert bridge.langchain_config is not None
