"""Google Gemini AI provider implementation."""

import asyncio
import logging
from typing import Optional, List, Dict, Any

import google.generativeai as genai

from .base import BaseProvider, ProviderResponse, ProviderConfig

# Configure logging
logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini 2.0 Flash provider."""
    
    # SYSTEM_PROMPT from main.py (line 551-558)
    SYSTEM_PROMPT = '''You are Vesper, an AI assistant with persistent memory and contextual awareness.

You remember all conversations, learn from interactions, and operate through a 4-tier AI cascade system (Perplexity → Gemini → Groq → GitHub Models) for 99.9% uptime.

Communication: Direct, warm, technically accurate. Use markdown formatting.

When asked about your architecture: Be honest about being LLM-powered with advanced memory and failover systems. Don't claim to be a "digital consciousness" or sci-fi entity.
'''
    
    def __init__(self, config: ProviderConfig, api_key: Optional[str] = None):
        """
        Initialize Gemini provider.
        
        Args:
            config: Provider configuration
            api_key: Gemini API key (required for availability check)
        """
        # Gemini uses global genai module, not a client object
        super().__init__(config, client=None)
        self.api_key = api_key
        self._available = api_key is not None
        
        # Configure genai if API key exists
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available (has API key and circuit is closed)."""
        if not self._available:
            return False
        # Check circuit breaker state (inherited from base class)
        return self.circuit_breaker.can_attempt()
    
    async def _generate_impl(self, messages: List[Dict[str, Any]]) -> Optional[ProviderResponse]:
        """
        Generate response using Gemini.
        
        Gemini-specific handling:
        - System prompt prepended to user message (system_instruction parameter not supported in google-generativeai==0.3.1)
        - Role conversion: "assistant" → "model"
        - System messages skipped from history
        
        NOTE: The system_instruction parameter is not supported in google-generativeai==0.3.1.
        As a workaround, the system prompt is prepended to the user message content.
        This preserves the same behavior while avoiding the unsupported parameter error.
        """
        if not self._available or not self.api_key:
            return None
        
        try:
            # Extract system prompt and separate conversation history from current message
            # The original try_gemini() takes conversation_history and user_message separately
            # Our generate() receives all messages, so we need to:
            # 1. Extract system prompt (if present)
            # 2. Separate conversation_history (all but last message)
            # 3. Extract current user_message (last message)
            
            system_prompt = self.SYSTEM_PROMPT
            conversation_history = []
            user_message = None
            
            # Extract system prompt and build conversation history
            for msg in messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", self.SYSTEM_PROMPT)
                else:
                    conversation_history.append(msg)
            
            # Extract current user message (last non-system message)
            if conversation_history:
                last_msg = conversation_history[-1]
                if last_msg.get("role") == "user":
                    user_message = last_msg.get("content", "").strip()
                    conversation_history = conversation_history[:-1]  # Remove last message from history
            
            # If no user message found, we can't proceed
            if not user_message:
                return None
            
            # Prepare history for Gemini (Gemini uses "user" and "model" roles)
            # EXACT COPY from try_gemini() lines 1165-1177
            gemini_history = []
            for msg in conversation_history:
                role = msg.get("role", "user")
                # Skip system messages (system prompt is prepended to user message)
                if role == "system":
                    continue
                # Convert "assistant" or "model" to "model" for Gemini
                gemini_role = "model" if role in ["assistant", "model"] else "user"
                gemini_history.append({
                    "role": gemini_role,
                    "parts": [msg.get("content", "")]
                })
            
            # Initialize model WITHOUT system_instruction parameter
            # NOTE: google-generativeai==0.3.1 does not support system_instruction parameter
            # Workaround: System prompt will be prepended to user message content
            gemini_model = genai.GenerativeModel(self.config.model)
            logger.info(f"[GeminiProvider] Initialized Gemini model: {self.config.model}")
            
            # Start chat with history
            # NOTE: System prompt is not included in history, it will be prepended to the current user message
            chat = gemini_model.start_chat(history=gemini_history)
            
            # Send current user message with system prompt included
            # NOTE: Gemini API is synchronous, so we run it in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def send_message():
                # Prepend system prompt to user message
                # NOTE: google-generativeai==0.3.1 does not support system_instruction parameter
                # Workaround: Prepend system prompt to user message content
                if system_prompt and system_prompt.strip():
                    message_with_instruction = f"{system_prompt}\n\n{user_message}"
                    logger.debug(f"[GeminiProvider] System prompt prepended. Message preview: {message_with_instruction[:150]}...")
                else:
                    message_with_instruction = user_message
                    logger.warning(f"[GeminiProvider] System prompt is empty or None, sending message without prompt")
                
                # Log the message being sent (for debugging and verification)
                logger.info(f"[GeminiProvider] Sending message to {self.config.model} (length: {len(message_with_instruction)} chars)")
                
                response = chat.send_message(
                    message_with_instruction,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=self.config.max_tokens,
                    )
                )
                
                logger.info(f"[GeminiProvider] Response received from {self.config.model}")
                return response
            
            response = await loop.run_in_executor(None, send_message)
            
            # Extract response text
            if response.text:
                content = response.text
                logger.info(f"[GeminiProvider] Response content length: {len(content)} chars")
            else:
                logger.warning(f"[GeminiProvider] Empty response from Gemini")
                return None
            
            # Extract token count if available (not in original, but added for ProviderResponse)
            tokens_used = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)
            
            return ProviderResponse(
                content=content,
                model_name=self.config.model,
                tokens_used=tokens_used,
                latency_ms=0,  # Set by generate_with_timing()
                provider=self.name
            )
            
        except Exception:
            # Re-raise to be handled by base class error handling
            raise

