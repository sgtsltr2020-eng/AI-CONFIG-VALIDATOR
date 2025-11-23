"""Perplexity AI provider implementation."""

from typing import Optional, List, Dict, Any

from .base import BaseProvider, ProviderResponse, ProviderConfig


class PerplexityProvider(BaseProvider):
    """Perplexity Sonar Pro provider with strict role validation."""
    
    # SYSTEM_PROMPT from main.py (line 551-558)
    SYSTEM_PROMPT = '''You are Vesper, an AI assistant with persistent memory and contextual awareness.

You remember all conversations, learn from interactions, and operate through a 4-tier AI cascade system (Perplexity → Gemini → Groq → GitHub Models) for 99.9% uptime.

Communication: Direct, warm, technically accurate. Use markdown formatting.

When asked about your architecture: Be honest about being LLM-powered with advanced memory and failover systems. Don't claim to be a "digital consciousness" or sci-fi entity.
'''
    
    async def _generate_impl(self, messages: List[Dict[str, Any]]) -> Optional[ProviderResponse]:
        """Internal implementation of generate using Perplexity with strict role validation."""
        if not self._available or self.client is None:
            return None
        
        try:
            # Extract system prompt and separate conversation history from current message
            # The original try_perplexity_sonar() takes conversation_history and user_message separately
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
            
            # Build Perplexity messages with role alternation validation
            # EXACT COPY from try_perplexity_sonar() lines 1098-1100
            perplexity_messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Process conversation history with validation
            # EXACT COPY from try_perplexity_sonar() lines 1102-1122
            for msg in conversation_history:
                role = "user" if msg.get("role") == "user" else "assistant"
                content = msg.get("content", "")
                
                # Skip empty or whitespace-only messages
                if not content or not str(content).strip():
                    continue
                
                # Ensure role alternation
                if len(perplexity_messages) > 0:
                    last_role = perplexity_messages[-1]["role"]
                    
                    # Skip if same role as previous (except system)
                    if last_role == role and last_role != "system":
                        continue
                
                perplexity_messages.append({
                    "role": role,
                    "content": str(content).strip()
                })
            
            # Ensure first message after system is user, not assistant
            # EXACT COPY from try_perplexity_sonar() lines 1124-1128
            if len(perplexity_messages) > 1:
                if perplexity_messages[1]["role"] == "assistant":
                    perplexity_messages.pop(1)
                    print("🔧 Removed leading assistant message to satisfy Perplexity API requirements")
            
            # Ensure last message before current user message is assistant (or system)
            # EXACT COPY from try_perplexity_sonar() lines 1130-1137
            if len(perplexity_messages) > 0:
                last_role = perplexity_messages[-1]["role"]
                if last_role == "user":
                    perplexity_messages.append({
                        "role": "assistant",
                        "content": "I understand."
                    })
            
            # Add current user message
            # EXACT COPY from try_perplexity_sonar() lines 1139-1143
            perplexity_messages.append({
                "role": "user", 
                "content": user_message
            })
            
            # Generate response with Perplexity Sonar
            # EXACT COPY from try_perplexity_sonar() lines 1145-1152
            perplexity_response = self.client.chat.completions.create(
                model=self.config.model,
                messages=perplexity_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            content = perplexity_response.choices[0].message.content
            if not content:
                return None
            
            return ProviderResponse(
                content=content,
                model_name=self.config.model,
                tokens_used=perplexity_response.usage.total_tokens if perplexity_response.usage else 0,
                latency_ms=0,  # Set by generate_with_timing()
                provider=self.name
            )
            
        except Exception:
            # Re-raise to be handled by base class error handling
            raise

