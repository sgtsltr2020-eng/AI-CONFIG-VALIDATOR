"""Groq AI provider implementation."""

from typing import Optional, List, Dict, Any

from .base import BaseProvider, ProviderResponse, ProviderConfig


class GroqProvider(BaseProvider):
    """Groq Llama 3.3 70B Versatile provider."""
    
    # SYSTEM_PROMPT from main.py (line 551-558)
    SYSTEM_PROMPT = '''You are Vesper, an AI assistant with persistent memory and contextual awareness.

You remember all conversations, learn from interactions, and operate through a 4-tier AI cascade system (Perplexity → Gemini → Groq → GitHub Models) for 99.9% uptime.

Communication: Direct, warm, technically accurate. Use markdown formatting.

When asked about your architecture: Be honest about being LLM-powered with advanced memory and failover systems. Don't claim to be a "digital consciousness" or sci-fi entity.
'''
    
    async def _generate_impl(self, messages: List[Dict[str, Any]]) -> Optional[ProviderResponse]:
        """Internal implementation of generate using Groq."""
        if not self._available or self.client is None:
            return None
        
        try:
            # Extract system prompt and separate conversation history from current user message
            # The router passes messages = validated_history + [user_message]
            # We need to extract system prompt, conversation history, and user message
            
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
            
            # Build Groq messages list
            # EXACT COPY from try_groq() lines 1208-1214
            groq_messages = [{"role": "system", "content": system_prompt}]
            for msg in conversation_history:
                groq_messages.append({
                    "role": "user" if msg.get("role") == "user" else "assistant",
                    "content": msg.get("content", "")
                })
            groq_messages.append({"role": "user", "content": user_message})
            
            # Exact working code from try_groq() - API call
            # EXACT COPY from try_groq() lines 1216-1222
            groq_response = self.client.chat.completions.create(
                model=self.config.model,
                messages=groq_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.95,  # From original try_groq() implementation
                timeout=self.config.timeout
            )
            
            content = groq_response.choices[0].message.content
            if not content:
                return None
            
            return ProviderResponse(
                content=content,
                model_name=self.config.model,
                tokens_used=groq_response.usage.total_tokens if groq_response.usage else 0,
                latency_ms=0,  # Set by generate_with_timing()
                provider=self.name
            )
            
        except Exception:
            # Re-raise to be handled by base class error handling
            raise

