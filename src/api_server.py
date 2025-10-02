"""
FastAPI Server for Dify-LangChain Integration

This module provides REST API endpoints for the Dify-LangChain integration.
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import get_settings, ConfigurationError
from src.dify_client import DifyClient, DifyAPIError
from src.langchain_config import LangChainConfig, DifyLangChainBridge

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dify-LangChain Integration API",
    description="REST API for Dify and LangChain integration",
    version="1.0.0"
)

# Global instances
dify_client = None
langchain_config = None
bridge = None


# Pydantic models for request/response
class MessageRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    conversation_id: Optional[str] = None
    response_mode: str = "streaming"


class MessageResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None


class LangChainRequest(BaseModel):
    message: str


class LangChainResponse(BaseModel):
    answer: str


class BridgeRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    mode: str = "preprocess"  # preprocess, postprocess, hybrid


class HealthResponse(BaseModel):
    status: str
    dify_configured: bool
    langchain_configured: bool


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global dify_client, langchain_config, bridge

    try:
        settings = get_settings()
        logger.info("Settings loaded successfully")

        # Initialize Dify client
        dify_client = DifyClient(
            api_key=settings.DIFY_API_KEY,
            api_url=settings.DIFY_API_URL
        )
        logger.info("Dify client initialized")

        # Initialize LangChain
        langchain_config = LangChainConfig(
            provider=settings.LLM_PROVIDER,
            model_name=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
        logger.info("LangChain configuration initialized")

        # Create bridge
        bridge = DifyLangChainBridge(dify_client, langchain_config)
        logger.info("Dify-LangChain bridge created")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Dify-LangChain Integration API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "dify_message": "POST /dify/message",
            "langchain_message": "POST /langchain/message",
            "bridge_preprocess": "POST /bridge/preprocess",
            "bridge_postprocess": "POST /bridge/postprocess",
            "bridge_hybrid": "POST /bridge/hybrid"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        dify_configured=dify_client is not None,
        langchain_configured=langchain_config is not None
    )


@app.post("/dify/message", response_model=MessageResponse)
async def send_dify_message(request: MessageRequest):
    """
    Send a message to Dify

    Args:
        request: Message request containing message and user_id

    Returns:
        Dify response with answer and conversation info
    """
    if dify_client is None:
        raise HTTPException(status_code=500, detail="Dify client not initialized")

    try:
        logger.info(f"Sending message to Dify: {request.message[:50]}...")
        response = dify_client.send_message(
            message=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            response_mode=request.response_mode
        )

        return MessageResponse(
            answer=response.get("answer", ""),
            conversation_id=response.get("conversation_id"),
            message_id=response.get("message_id")
        )

    except DifyAPIError as e:
        logger.error(f"Dify API error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/langchain/message", response_model=LangChainResponse)
async def send_langchain_message(request: LangChainRequest):
    """
    Send a message to LangChain

    Args:
        request: Message request

    Returns:
        LangChain response
    """
    if langchain_config is None:
        raise HTTPException(status_code=500, detail="LangChain not initialized")

    try:
        logger.info(f"Sending message to LangChain: {request.message[:50]}...")

        # Create a simple chain
        chain = langchain_config.create_simple_chain(
            template="Answer the following question: {message}",
            input_variables=["message"]
        )

        response = chain.run(message=request.message)

        return LangChainResponse(answer=response)

    except Exception as e:
        logger.error(f"LangChain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/preprocess")
async def bridge_preprocess(request: BridgeRequest):
    """
    Preprocess message with LangChain, then send to Dify

    Args:
        request: Bridge request

    Returns:
        Combined response
    """
    if bridge is None:
        raise HTTPException(status_code=500, detail="Bridge not initialized")

    try:
        # Create preprocessing chain
        preprocessing_chain = langchain_config.create_simple_chain(
            template="Enhance and clarify this question: {message}",
            input_variables=["message"]
        )

        result = bridge.process_with_langchain_then_dify(
            message=request.message,
            user_id=request.user_id,
            preprocessing_chain=preprocessing_chain
        )

        return result

    except Exception as e:
        logger.error(f"Bridge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/postprocess")
async def bridge_postprocess(request: BridgeRequest):
    """
    Send message to Dify, then postprocess with LangChain

    Args:
        request: Bridge request

    Returns:
        Combined response
    """
    if bridge is None:
        raise HTTPException(status_code=500, detail="Bridge not initialized")

    try:
        # Create postprocessing chain
        postprocessing_chain = langchain_config.create_simple_chain(
            template="Summarize this response in a concise way: {response}",
            input_variables=["response"]
        )

        result = bridge.process_with_dify_then_langchain(
            message=request.message,
            user_id=request.user_id,
            postprocessing_chain=postprocessing_chain
        )

        return result

    except Exception as e:
        logger.error(f"Bridge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bridge/hybrid")
async def bridge_hybrid(request: BridgeRequest):
    """
    Get responses from both Dify and LangChain

    Args:
        request: Bridge request

    Returns:
        Hybrid response from both systems
    """
    if bridge is None:
        raise HTTPException(status_code=500, detail="Bridge not initialized")

    try:
        # Create LangChain chain
        langchain_chain = langchain_config.create_simple_chain(
            template="Answer this question: {message}",
            input_variables=["message"]
        )

        result = bridge.hybrid_response(
            message=request.message,
            user_id=request.user_id,
            langchain_chain=langchain_chain,
            combine_strategy="concatenate"
        )

        return result

    except Exception as e:
        logger.error(f"Bridge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api_server:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )
