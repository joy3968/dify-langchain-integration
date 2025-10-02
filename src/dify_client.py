"""
Dify API Client Module

This module provides a client for interacting with the Dify API.
"""

import logging
import os
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DifyClientError(Exception):
    """Base exception for Dify client errors"""
    pass


class DifyAuthenticationError(DifyClientError):
    """Exception raised for authentication errors"""
    pass


class DifyAPIError(DifyClientError):
    """Exception raised for API errors"""
    pass


class DifyClient:
    """Client for interacting with Dify API"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize Dify client

        Args:
            api_key: Dify API key (defaults to DIFY_API_KEY env var)
            api_url: Dify API URL (defaults to DIFY_API_URL env var)
        """
        self.api_key = api_key or os.getenv('DIFY_API_KEY')
        self.api_url = (api_url or os.getenv('DIFY_API_URL', 'https://api.dify.ai/v1')).rstrip('/')

        if not self.api_key:
            raise DifyAuthenticationError(
                "API key not found. Please set DIFY_API_KEY environment variable or pass api_key parameter."
            )

        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

        logger.info(f"Dify client initialized with API URL: {self.api_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Dify API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response

        Returns:
            API response as dictionary

        Raises:
            DifyAPIError: If API request fails
        """
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        try:
            logger.debug(f"Making {method} request to {url}")
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=30,
                stream=stream
            )

            response.raise_for_status()

            # Handle streaming response
            if stream:
                result = {"answer": "", "events": []}
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                import json
                                event_data = json.loads(line_str[6:])
                                result["events"].append(event_data)

                                # Accumulate answer text
                                if event_data.get('event') == 'agent_message':
                                    result["answer"] += event_data.get('answer', '')
                                elif event_data.get('event') == 'message':
                                    result["answer"] = event_data.get('answer', '')
                                    result["conversation_id"] = event_data.get('conversation_id', '')
                                    result["message_id"] = event_data.get('id', '')
                            except json.JSONDecodeError:
                                continue
                logger.debug(f"Streaming request successful")
                return result
            else:
                result = response.json()
                logger.debug(f"Request successful: {response.status_code}")
                return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            error_msg = f"API request failed with status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {response.text}"
            raise DifyAPIError(error_msg) from e

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise DifyAPIError(f"Failed to connect to Dify API at {url}") from e

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout: {e}")
            raise DifyAPIError("Request to Dify API timed out") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise DifyAPIError(f"Request failed: {str(e)}") from e

    def send_message(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        response_mode: str = "blocking"
    ) -> Dict[str, Any]:
        """
        Send a message to Dify API

        Args:
            message: The message to send
            user_id: Unique identifier for the user
            conversation_id: Optional conversation ID for continuing a conversation
            inputs: Optional additional inputs for the conversation
            response_mode: Response mode - "blocking" or "streaming"

        Returns:
            API response containing the assistant's reply

        Raises:
            DifyAPIError: If the request fails
        """
        logger.info(f"Sending message for user {user_id}")

        data = {
            "query": message,
            "user": user_id,
            "response_mode": response_mode,
            "inputs": inputs or {}
        }

        if conversation_id:
            data["conversation_id"] = conversation_id

        try:
            is_streaming = response_mode == "streaming"
            response = self._make_request("POST", "/chat-messages", data=data, stream=is_streaming)
            logger.info("Message sent successfully")
            return response

        except DifyAPIError as e:
            logger.error(f"Failed to send message: {e}")
            raise

    def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Retrieve messages from a conversation

        Args:
            conversation_id: The conversation ID
            user_id: Unique identifier for the user
            limit: Maximum number of messages to retrieve

        Returns:
            API response containing conversation messages

        Raises:
            DifyAPIError: If the request fails
        """
        logger.info(f"Retrieving messages for conversation {conversation_id}")

        params = {
            "user": user_id,
            "limit": limit
        }

        try:
            response = self._make_request(
                "GET",
                f"/messages/{conversation_id}",
                params=params
            )
            logger.info(f"Retrieved {len(response.get('data', []))} messages")
            return response

        except DifyAPIError as e:
            logger.error(f"Failed to retrieve messages: {e}")
            raise

    def get_conversations(
        self,
        user_id: str,
        limit: int = 20,
        last_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get list of conversations for a user

        Args:
            user_id: Unique identifier for the user
            limit: Maximum number of conversations to retrieve
            last_id: ID of last conversation for pagination

        Returns:
            API response containing list of conversations

        Raises:
            DifyAPIError: If the request fails
        """
        logger.info(f"Retrieving conversations for user {user_id}")

        params = {
            "user": user_id,
            "limit": limit
        }

        if last_id:
            params["last_id"] = last_id

        try:
            response = self._make_request("GET", "/conversations", params=params)
            logger.info(f"Retrieved {len(response.get('data', []))} conversations")
            return response

        except DifyAPIError as e:
            logger.error(f"Failed to retrieve conversations: {e}")
            raise

    def close(self):
        """Close the session"""
        self.session.close()
        logger.info("Dify client session closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
