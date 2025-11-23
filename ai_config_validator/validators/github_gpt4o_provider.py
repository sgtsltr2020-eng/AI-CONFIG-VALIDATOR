"""GitHub Models provider implementation."""

from typing import Optional, List, Dict, Any

from .base import BaseProvider, ProviderResponse, ProviderConfig


class GitHubGPT4oProvider(BaseProvider):
    """GitHub Models GPT-4o-mini provider."""
    
    # SYSTEM_PROMPT from main.py (line 551-558)
    SYSTEM_PROMPT = '''You are Vesper, an AI assistant with persistent memory and contextual awareness.

You remember all conversations, learn from interactions, and operate through a 4-tier AI cascade system (Perplexity → Gemini → Groq → GitHub Models) for 99.9% uptime.

Communication: Direct, warm, technically accurate. Use markdown formatting.

When asked about your architecture: Be honest about being LLM-powered with advanced memory and failover systems. Don't claim to be a "digital consciousness" or sci-fi entity.
'''
    
    async def _generate_impl(self, messages: List[Dict[str, Any]]) -> Optional[ProviderResponse]:
        """Internal implementation of generate using GitHub Models."""
        if not self._available or self.client is None:
            return None
        
        try:
            # Extract system prompt and separate conversation history from current message
            # The original try_github_models() takes conversation_history and user_message separately
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
            
            # Build GitHub messages list
            # EXACT COPY from try_github_models() lines 1239-1245
            github_messages = [{"role": "system", "content": system_prompt}]
            for msg in conversation_history:
                github_messages.append({
                    "role": "user" if msg.get("role") == "user" else "assistant",
                    "content": msg.get("content", "")
                })
            github_messages.append({"role": "user", "content": user_message})
            
            # Generate response with GitHub Models
            # EXACT COPY from try_github_models() lines 1247-1253
            github_response = self.client.chat.completions.create(
                model=self.config.model,
                messages=github_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            # Extract response content
            # EXACT COPY from try_github_models() lines 1255-1256
            if github_response.choices[0].message.content:
                content = github_response.choices[0].message.content
            else:
                return None
            
            # Extract token count if available (not in original, but added for ProviderResponse)
            tokens_used = 0
            if hasattr(github_response, 'usage') and github_response.usage:
                tokens_used = getattr(github_response.usage, 'total_tokens', 0)
            
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

