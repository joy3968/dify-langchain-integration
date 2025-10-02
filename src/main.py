"""
Main Application Entry Point

This module provides CLI and programmatic interfaces for the Dify-LangChain integration.
"""

import sys
import logging
from typing import Optional
import argparse

from config.settings import get_settings, ConfigurationError
from src.dify_client import DifyClient, DifyAPIError
from src.langchain_config import (
    LangChainConfig,
    DifyLangChainBridge,
    create_bridge
)

# Configure logging
logger = logging.getLogger(__name__)


class DifyLangChainApp:
    """Main application class for Dify-LangChain integration"""

    def __init__(self):
        """Initialize the application"""
        try:
            self.settings = get_settings()
            logger.info("Settings loaded successfully")

            # Initialize Dify client
            self.dify_client = DifyClient(
                api_key=self.settings.DIFY_API_KEY,
                api_url=self.settings.DIFY_API_URL
            )
            logger.info("Dify client initialized")

            # Initialize LangChain
            self.langchain_config = LangChainConfig(
                provider=self.settings.LLM_PROVIDER,
                model_name=self.settings.LLM_MODEL_NAME,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS
            )
            logger.info("LangChain configuration initialized")

            # Create bridge
            self.bridge = DifyLangChainBridge(
                self.dify_client,
                self.langchain_config
            )
            logger.info("Dify-LangChain bridge created")

        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def send_to_dify(self, message: str, user_id: str = "default_user", response_mode: str = "streaming") -> dict:
        """
        Send a message to Dify

        Args:
            message: Message to send
            user_id: User identifier
            response_mode: Response mode - "blocking" or "streaming"

        Returns:
            Dify response
        """
        try:
            logger.info(f"Sending message to Dify: {message[:50]}...")
            response = self.dify_client.send_message(
                message=message,
                user_id=user_id,
                response_mode=response_mode
            )
            logger.info("Received response from Dify")
            return response
        except DifyAPIError as e:
            logger.error(f"Dify API error: {e}")
            raise

    def send_to_langchain(self, message: str) -> str:
        """
        Send a message to LangChain

        Args:
            message: Message to send

        Returns:
            LangChain response
        """
        try:
            logger.info(f"Sending message to LangChain: {message[:50]}...")

            # Create a simple chain
            chain = self.langchain_config.create_simple_chain(
                template="Answer the following question: {message}",
                input_variables=["message"]
            )

            response = chain.run(message=message)
            logger.info("Received response from LangChain")
            return response
        except Exception as e:
            logger.error(f"LangChain error: {e}")
            raise

    def interactive_conversation(self, user_id: str = "default_user"):
        """
        Start an interactive conversation using LangChain's ConversationChain

        Args:
            user_id: User identifier
        """
        print("\n=== Interactive Conversation Mode ===")
        print("Using LangChain ConversationChain with memory")
        print("Type 'quit' or 'exit' to end the conversation\n")

        # Create conversation chain with memory
        chain = self.langchain_config.create_conversation_chain(
            memory_type=self.settings.MEMORY_TYPE,
            window_size=self.settings.MEMORY_WINDOW_SIZE
        )

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                # Get response from LangChain
                response = chain.run(user_input)
                print(f"Assistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print(f"Error: {e}\n")

    def demo_basic_usage(self):
        """Demonstrate basic usage of Dify and LangChain"""
        print("\n=== Basic Usage Demo ===\n")

        # Example 1: Send message to Dify
        print("1. Sending message to Dify:")
        try:
            dify_response = self.send_to_dify(
                "What is artificial intelligence?",
                user_id="demo_user"
            )
            print(f"   Response: {dify_response.get('answer', 'No answer')[:200]}...\n")
        except Exception as e:
            print(f"   Error: {e}\n")

        # Example 2: Send message to LangChain
        print("2. Sending message to LangChain:")
        try:
            langchain_response = self.send_to_langchain(
                "What is artificial intelligence?"
            )
            print(f"   Response: {langchain_response[:200]}...\n")
        except Exception as e:
            print(f"   Error: {e}\n")

        # Example 3: Use bridge for preprocessing
        print("3. Using bridge (LangChain preprocessing → Dify):")
        try:
            # Create preprocessing chain
            preprocess_chain = self.langchain_config.create_simple_chain(
                template="Enhance this question to be more specific: {message}",
                input_variables=["message"]
            )

            result = self.bridge.process_with_langchain_then_dify(
                message="Tell me about AI",
                user_id="demo_user",
                preprocessing_chain=preprocess_chain
            )

            print(f"   Original: {result['original_message']}")
            print(f"   Enhanced: {result['processed_message']}")
            print(f"   Dify Response: {result['dify_response'].get('answer', '')[:200]}...\n")
        except Exception as e:
            print(f"   Error: {e}\n")

    def demo_chains(self):
        """Demonstrate different LangChain chain types"""
        print("\n=== LangChain Chains Demo ===\n")

        # Example 1: Simple LLMChain
        print("1. Simple LLMChain:")
        try:
            chain = self.langchain_config.create_simple_chain(
                template="What is a good name for a company that makes {product}?",
                input_variables=["product"]
            )
            result = chain.run(product="eco-friendly water bottles")
            print(f"   Result: {result}\n")
        except Exception as e:
            print(f"   Error: {e}\n")

        # Example 2: Chat-style prompt
        print("2. Chat Prompt Chain (Translation):")
        try:
            chain = self.langchain_config.create_chat_prompt_chain(
                system_template="You are a translator. Translate from {input_language} to {output_language}.",
                human_template="{text}"
            )
            result = chain.run(
                input_language="English",
                output_language="Spanish",
                text="Hello, how are you today?"
            )
            print(f"   Result: {result}\n")
        except Exception as e:
            print(f"   Error: {e}\n")

        # Example 3: Conversation with memory
        print("3. Conversation Chain with Memory:")
        try:
            chain = self.langchain_config.create_conversation_chain()

            response1 = chain.run("Hi, my name is Alice and I love programming.")
            print(f"   User: Hi, my name is Alice and I love programming.")
            print(f"   Assistant: {response1[:150]}...")

            response2 = chain.run("What's my name and what do I love?")
            print(f"   User: What's my name and what do I love?")
            print(f"   Assistant: {response2[:150]}...\n")
        except Exception as e:
            print(f"   Error: {e}\n")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.dify_client.close()
            logger.info("Application cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(
        description="Dify-LangChain Integration Application"
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["demo", "chat", "dify", "langchain", "chains"],
        default="demo",
        help="Execution mode: demo (show examples), chat (interactive), dify (send to Dify), langchain (send to LangChain), chains (demo chains)"
    )

    parser.add_argument(
        "-m", "--message",
        type=str,
        help="Message to send (for dify/langchain modes)"
    )

    parser.add_argument(
        "-u", "--user-id",
        type=str,
        default="default_user",
        help="User ID for Dify requests"
    )

    args = parser.parse_args()

    try:
        # Initialize application
        print("Initializing Dify-LangChain application...")
        app = DifyLangChainApp()
        print("Initialization complete!\n")

        # Execute based on mode
        if args.mode == "demo":
            app.demo_basic_usage()

        elif args.mode == "chains":
            app.demo_chains()

        elif args.mode == "chat":
            app.interactive_conversation(user_id=args.user_id)

        elif args.mode == "dify":
            if not args.message:
                print("Error: --message is required for 'dify' mode")
                sys.exit(1)

            response = app.send_to_dify(args.message, user_id=args.user_id)
            print(f"\nDify Response:")
            print(f"Answer: {response.get('answer', 'No answer')}")
            print(f"Conversation ID: {response.get('conversation_id', 'N/A')}")

        elif args.mode == "langchain":
            if not args.message:
                print("Error: --message is required for 'langchain' mode")
                sys.exit(1)

            response = app.send_to_langchain(args.message)
            print(f"\nLangChain Response:")
            print(response)

        # Cleanup
        app.cleanup()

    except ConfigurationError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease ensure your .env file is properly configured.")
        print("Copy .env.example to .env and add your API keys.")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
