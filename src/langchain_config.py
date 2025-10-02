"""
LangChain Configuration Module

This module sets up LangChain with various LLM providers and integrates with Dify.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

from src.dify_client import DifyClient, DifyAPIError

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class LangChainConfig:
    """Configuration and setup for LangChain with various providers"""

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LangChain configuration

        Args:
            provider: LLM provider (currently supports 'openai')
            model_name: Model name (e.g., 'gpt-3.5-turbo', 'gpt-4')
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set default model names based on provider
        if model_name is None:
            self.model_name = "gpt-3.5-turbo" if provider == "openai" else "default"
        else:
            self.model_name = model_name

        self.llm = self._initialize_llm()
        logger.info(f"LangChain initialized with provider: {provider}, model: {self.model_name}")

    def _initialize_llm(self):
        """
        Initialize the LLM based on the provider

        Returns:
            Initialized LLM instance

        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            # Use ChatOpenAI for chat models
            if self.model_name.startswith("gpt-3.5") or self.model_name.startswith("gpt-4"):
                return ChatOpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=api_key
                )
            else:
                return OpenAI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    openai_api_key=api_key
                )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def create_simple_chain(
        self,
        template: str,
        input_variables: Optional[List[str]] = None
    ) -> LLMChain:
        """
        Create a simple LLMChain with a custom prompt template

        Args:
            template: The prompt template string
            input_variables: List of input variable names

        Returns:
            Configured LLMChain

        Example:
            template = "What is a good name for a company that makes {product}?"
            chain = config.create_simple_chain(template, ["product"])
            result = chain.run(product="colorful socks")
        """
        if input_variables is None:
            # Try to infer input variables from template
            import re
            input_variables = re.findall(r'\{(\w+)\}', template)

        prompt = PromptTemplate(
            template=template,
            input_variables=input_variables
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info(f"Created simple chain with variables: {input_variables}")
        return chain

    def create_conversation_chain(
        self,
        memory_type: str = "buffer",
        window_size: int = 5,
        system_message: Optional[str] = None
    ) -> ConversationChain:
        """
        Create a ConversationChain with memory

        Args:
            memory_type: Type of memory ('buffer' or 'window')
            window_size: Number of previous messages to remember (for window type)
            system_message: Optional system message to set context

        Returns:
            Configured ConversationChain

        Example:
            chain = config.create_conversation_chain()
            response1 = chain.run("Hi, my name is Alice")
            response2 = chain.run("What's my name?")
        """
        if memory_type == "buffer":
            memory = ConversationBufferMemory()
        elif memory_type == "window":
            memory = ConversationBufferWindowMemory(k=window_size)
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        # Create conversation chain
        chain = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=True
        )

        logger.info(f"Created conversation chain with {memory_type} memory")
        return chain

    def create_chat_prompt_chain(
        self,
        system_template: str,
        human_template: str
    ) -> LLMChain:
        """
        Create a chain with chat-style prompts (system + human messages)

        Args:
            system_template: Template for system message
            human_template: Template for human message

        Returns:
            Configured LLMChain with chat prompt

        Example:
            system = "You are a helpful assistant that translates {input_language} to {output_language}."
            human = "{text}"
            chain = config.create_chat_prompt_chain(system, human)
            result = chain.run(
                input_language="English",
                output_language="French",
                text="Hello, how are you?"
            )
        """
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])

        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        logger.info("Created chat prompt chain")
        return chain


class DifyLangChainBridge:
    """Bridge between Dify and LangChain for hybrid workflows"""

    def __init__(
        self,
        dify_client: DifyClient,
        langchain_config: LangChainConfig
    ):
        """
        Initialize the bridge between Dify and LangChain

        Args:
            dify_client: Initialized DifyClient instance
            langchain_config: Initialized LangChainConfig instance
        """
        self.dify_client = dify_client
        self.langchain_config = langchain_config
        logger.info("Dify-LangChain bridge initialized")

    def process_with_langchain_then_dify(
        self,
        message: str,
        user_id: str,
        preprocessing_chain: LLMChain,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Preprocess message with LangChain, then send to Dify

        Args:
            message: Original message
            user_id: User identifier
            preprocessing_chain: LangChain chain for preprocessing
            conversation_id: Optional Dify conversation ID

        Returns:
            Dify API response

        Example workflow: Use LangChain to enhance/format query, then send to Dify
        """
        try:
            # Preprocess with LangChain
            logger.info("Preprocessing message with LangChain")
            processed_message = preprocessing_chain.run(message=message)
            logger.debug(f"Processed message: {processed_message}")

            # Send to Dify
            logger.info("Sending processed message to Dify")
            response = self.dify_client.send_message(
                message=processed_message,
                user_id=user_id,
                conversation_id=conversation_id
            )

            return {
                "original_message": message,
                "processed_message": processed_message,
                "dify_response": response
            }

        except DifyAPIError as e:
            logger.error(f"Dify API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in LangChain-Dify processing: {e}")
            raise

    def process_with_dify_then_langchain(
        self,
        message: str,
        user_id: str,
        postprocessing_chain: LLMChain,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send message to Dify, then postprocess response with LangChain

        Args:
            message: User message
            user_id: User identifier
            postprocessing_chain: LangChain chain for postprocessing
            conversation_id: Optional Dify conversation ID

        Returns:
            Combined response with both Dify and LangChain outputs

        Example workflow: Get Dify response, then use LangChain to summarize/format
        """
        try:
            # Get response from Dify
            logger.info("Sending message to Dify")
            dify_response = self.dify_client.send_message(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id
            )

            # Extract answer from Dify response
            dify_answer = dify_response.get("answer", "")
            logger.debug(f"Dify answer: {dify_answer}")

            # Postprocess with LangChain
            logger.info("Postprocessing Dify response with LangChain")
            final_response = postprocessing_chain.run(response=dify_answer)

            return {
                "original_message": message,
                "dify_response": dify_response,
                "dify_answer": dify_answer,
                "processed_response": final_response
            }

        except DifyAPIError as e:
            logger.error(f"Dify API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in Dify-LangChain processing: {e}")
            raise

    def hybrid_response(
        self,
        message: str,
        user_id: str,
        langchain_chain: LLMChain,
        conversation_id: Optional[str] = None,
        combine_strategy: str = "concatenate"
    ) -> Dict[str, Any]:
        """
        Get responses from both Dify and LangChain, then combine them

        Args:
            message: User message
            user_id: User identifier
            langchain_chain: LangChain chain to use
            conversation_id: Optional Dify conversation ID
            combine_strategy: How to combine responses ('concatenate', 'choose_best', etc.)

        Returns:
            Combined response from both systems
        """
        try:
            # Get response from Dify
            logger.info("Getting response from Dify")
            dify_response = self.dify_client.send_message(
                message=message,
                user_id=user_id,
                conversation_id=conversation_id
            )
            dify_answer = dify_response.get("answer", "")

            # Get response from LangChain
            logger.info("Getting response from LangChain")
            langchain_answer = langchain_chain.run(message=message)

            # Combine responses based on strategy
            if combine_strategy == "concatenate":
                combined = f"Dify: {dify_answer}\n\nLangChain: {langchain_answer}"
            else:
                combined = dify_answer  # Default to Dify

            return {
                "message": message,
                "dify_answer": dify_answer,
                "langchain_answer": langchain_answer,
                "combined_answer": combined,
                "strategy": combine_strategy
            }

        except Exception as e:
            logger.error(f"Error in hybrid response: {e}")
            raise


# Example usage and helper functions
def create_default_config() -> LangChainConfig:
    """Create a default LangChain configuration"""
    return LangChainConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )


def create_bridge(
    dify_api_key: Optional[str] = None,
    langchain_provider: str = "openai"
) -> DifyLangChainBridge:
    """
    Create a ready-to-use Dify-LangChain bridge

    Args:
        dify_api_key: Optional Dify API key (uses env var if not provided)
        langchain_provider: LLM provider for LangChain

    Returns:
        Configured DifyLangChainBridge instance
    """
    dify_client = DifyClient(api_key=dify_api_key)
    langchain_config = LangChainConfig(provider=langchain_provider)
    return DifyLangChainBridge(dify_client, langchain_config)
