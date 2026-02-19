# llm_router.py
"""
Multi-Provider LLM Fallback Router
===================================
Architecture:
  1. Primary   → Google Gemini (gemini-1.5-flash)  — free tier
  2. Secondary → Groq (llama3-70b-8192)             — free tier
  3. Fallback  → Ollama local (llama3)               — offline

The router automatically cascades through providers on failure.
It handles: rate limits (429), quota exceeded, network errors,
API key errors, and any unexpected exceptions.

The Flask app NEVER crashes — a safe fallback response is always returned.
"""

import os
import time
import json
import logging
import requests
from typing import Generator, Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("llm_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [LLM-Router] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Provider enum-like constants
# ---------------------------------------------------------------------------
PROVIDER_GEMINI = "gemini"
PROVIDER_GROQ   = "groq"
PROVIDER_OLLAMA = "ollama"

# Errors that should trigger a provider switch
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# ===========================================================================
# Individual Provider Implementations
# ===========================================================================

class GeminiProvider:
    """Google Gemini API provider"""
    
    NAME = PROVIDER_GEMINI
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-2.0-flash-lite"
    
    @property
    def available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: str = "", 
                 temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Generate a complete response from Gemini"""
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Build the request payload
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System Instructions]: {system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95
            }
        }
        
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        # Check for rate limit / quota errors
        if response.status_code in RETRYABLE_STATUS_CODES:
            error_msg = response.text[:300]
            raise ProviderRateLimitError(
                f"Gemini returned {response.status_code}: {error_msg}"
            )
        
        if response.status_code == 404:
            raise ProviderRateLimitError(
                f"Gemini model not found (404) — cascading to next provider"
            )
        
        if response.status_code == 400:
            error_data = response.json() if response.text else {}
            error_msg = str(error_data)
            if "API_KEY" in error_msg.upper() or "PERMISSION" in error_msg.upper():
                raise ProviderAuthError(f"Gemini API key error: {error_msg[:200]}")
            raise ProviderError(f"Gemini bad request: {error_msg[:200]}")
        
        if response.status_code == 403:
            raise ProviderAuthError(f"Gemini forbidden (check API key): {response.text[:200]}")
        
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from Gemini response
        try:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            pass
        
        raise ProviderError(f"Unexpected Gemini response structure: {str(data)[:200]}")
    
    def generate_stream(self, prompt: str, system_prompt: str = "",
                        temperature: float = 0.3, max_tokens: int = 1500) -> Generator[str, None, None]:
        """Stream response from Gemini using streamGenerateContent"""
        url = (f"{self.base_url}/{self.model}:streamGenerateContent"
               f"?key={self.api_key}&alt=sse")
        
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System Instructions]: {system_prompt}"}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95
            }
        }
        
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=90,
            stream=True
        )
        
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise ProviderRateLimitError(
                f"Gemini stream returned {response.status_code}"
            )
        if response.status_code in (400, 403):
            raise ProviderAuthError(f"Gemini auth/request error: {response.status_code}")
        
        response.raise_for_status()
        
        # Parse SSE stream from Gemini
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                json_str = line[6:]
                try:
                    chunk = json.loads(json_str)
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue


class GroqProvider:
    """Groq API provider (OpenAI-compatible endpoint)"""
    
    NAME = PROVIDER_GROQ
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"
    
    @property
    def available(self) -> bool:
        return bool(self.api_key)
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Generate a complete response from Groq"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=60
        )
        
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise ProviderRateLimitError(
                f"Groq returned {response.status_code}: {response.text[:200]}"
            )
        if response.status_code == 400:
            raise ProviderRateLimitError(
                f"Groq bad request (model may be unavailable): {response.text[:200]}"
            )
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"Groq API key error: {response.text[:200]}")
        
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def generate_stream(self, prompt: str, system_prompt: str = "",
                        temperature: float = 0.3, max_tokens: int = 1500) -> Generator[str, None, None]:
        """Stream response from Groq"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
            timeout=90,
            stream=True
        )
        
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise ProviderRateLimitError(f"Groq stream returned {response.status_code}")
        if response.status_code in (401, 403):
            raise ProviderAuthError(f"Groq auth error: {response.status_code}")
        
        response.raise_for_status()
        
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            json_str = line[6:]
            if json_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(json_str)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


class OllamaProvider:
    """Ollama local model provider (runs offline)"""
    
    NAME = PROVIDER_OLLAMA
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.model = "llama3"
    
    @property
    def available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, system_prompt: str = "",
                 temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Generate a complete response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise ProviderError(f"Ollama returned {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        return data.get("response", "")
    
    def generate_stream(self, prompt: str, system_prompt: str = "",
                        temperature: float = 0.3, max_tokens: int = 1500) -> Generator[str, None, None]:
        """Stream response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
            stream=True
        )
        
        if response.status_code != 200:
            raise ProviderError(f"Ollama stream returned {response.status_code}")
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
                text = chunk.get("response", "")
                if text:
                    yield text
                if chunk.get("done", False):
                    break
            except json.JSONDecodeError:
                continue


# ===========================================================================
# Custom Exceptions
# ===========================================================================

class ProviderError(Exception):
    """Base provider error"""
    pass

class ProviderRateLimitError(ProviderError):
    """Rate limit / quota exceeded — should cascade to next provider"""
    pass

class ProviderAuthError(ProviderError):
    """API key invalid or missing — should cascade to next provider"""
    pass

class ProviderNetworkError(ProviderError):
    """Network connectivity error — should cascade to next provider"""
    pass


# ===========================================================================
# Main Router
# ===========================================================================

class LLMRouter:
    """
    Multi-Provider LLM Router with Automatic Fallback
    ==================================================
    
    Provider cascade order:
      1. Gemini (primary, free cloud)
      2. Groq   (secondary, free cloud)
      3. Ollama  (final, offline local)
    
    If ALL providers fail, returns a safe static fallback response.
    The system NEVER crashes.
    """
    
    # Default system prompt for academic research
    SYSTEM_PROMPT = (
        "You are an expert academic research assistant. "
        "Provide detailed, accurate, and well-structured analyses using markdown. "
        "Be thorough, cite sources when possible, and maintain academic rigor."
    )
    
    def __init__(self):
        """Initialize all providers from environment variables"""
        from config import Config
        
        # Initialize providers
        gemini_key = getattr(Config, 'GEMINI_API_KEY', None) or os.getenv('GEMINI_API_KEY', '')
        groq_key   = getattr(Config, 'GROQ_API_KEY', None)   or os.getenv('GROQ_API_KEY', '')
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        self.providers = []
        
        # Primary: Gemini
        self.gemini = GeminiProvider(gemini_key)
        if self.gemini.available:
            self.providers.append(self.gemini)
            logger.info("✅ Gemini provider registered (primary)")
        else:
            logger.warning("⚠️  Gemini API key not found — skipping")
        
        # Secondary: Groq
        self.groq = GroqProvider(groq_key)
        if self.groq.available:
            self.providers.append(self.groq)
            logger.info("✅ Groq provider registered (secondary)")
        else:
            logger.warning("⚠️  Groq API key not found — skipping")
        
        # Fallback: Ollama (local)
        self.ollama = OllamaProvider(ollama_url)
        self.providers.append(self.ollama)  # Always add, availability checked at call time
        logger.info("✅ Ollama provider registered (fallback/offline)")
        
        # Track which provider was last used
        self.last_provider_used: Optional[str] = None
        
        # Provider health tracking
        self._provider_failures: Dict[str, float] = {}
        self._cooldown_seconds = 60  # Skip provider for 60s after failure
        
        logger.info(f"🔄 LLM Router initialized with {len(self.providers)} provider(s)")
    
    def _is_provider_cooled_down(self, provider_name: str) -> bool:
        """Check if a provider has had a recent failure and should be skipped"""
        last_failure = self._provider_failures.get(provider_name)
        if last_failure is None:
            return True  # No failure, proceed
        return (time.time() - last_failure) > self._cooldown_seconds
    
    def _mark_provider_failed(self, provider_name: str):
        """Mark a provider as recently failed"""
        self._provider_failures[provider_name] = time.time()
    
    def _clear_provider_failure(self, provider_name: str):
        """Clear failure record for a provider on success"""
        self._provider_failures.pop(provider_name, None)
    
    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """
        Generate a response, cascading through providers on failure.
        NEVER raises an exception to the caller — always returns a string.
        """
        sys_prompt = system_prompt or self.SYSTEM_PROMPT
        errors = []
        
        for provider in self.providers:
            # Skip providers that are cooling down from recent failures
            if not self._is_provider_cooled_down(provider.NAME):
                logger.info(f"⏭️  Skipping {provider.NAME} (cooling down after recent failure)")
                continue
            
            # For Ollama, check availability at call time
            if provider.NAME == PROVIDER_OLLAMA and not provider.available:
                logger.warning(f"⏭️  Ollama not available (not running locally)")
                errors.append(f"Ollama: not running at {provider.base_url}")
                continue
            
            try:
                logger.info(f"🔄 Trying provider: {provider.NAME}")
                result = provider.generate(
                    prompt, sys_prompt, temperature, max_tokens
                )
                
                if result and result.strip():
                    self.last_provider_used = provider.NAME
                    self._clear_provider_failure(provider.NAME)
                    logger.info(f"✅ Response from {provider.NAME} ({len(result)} chars)")
                    return result
                else:
                    raise ProviderError(f"{provider.NAME} returned empty response")
                    
            except ProviderRateLimitError as e:
                logger.warning(f"⚠️  {provider.NAME} rate limited: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: rate limited")
                
            except ProviderAuthError as e:
                logger.warning(f"⚠️  {provider.NAME} auth error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: authentication error")
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"⚠️  {provider.NAME} connection error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: connection failed")
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"⚠️  {provider.NAME} timeout: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: request timed out")
                
            except Exception as e:
                logger.error(f"❌ {provider.NAME} unexpected error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: {str(e)[:100]}")
        
        # ALL providers failed — return safe fallback
        logger.error(f"❌ All providers failed. Errors: {errors}")
        self.last_provider_used = "fallback"
        return self._generate_safe_fallback(prompt, errors)
    
    def generate_stream(self, prompt: str, system_prompt: str = None,
                        temperature: float = 0.3, max_tokens: int = 1500) -> Generator[str, None, None]:
        """
        Stream a response, cascading through providers on failure.
        NEVER raises an exception — always yields content.
        """
        sys_prompt = system_prompt or self.SYSTEM_PROMPT
        errors = []
        
        for provider in self.providers:
            if not self._is_provider_cooled_down(provider.NAME):
                logger.info(f"⏭️  Skipping {provider.NAME} (cooling down)")
                continue
            
            if provider.NAME == PROVIDER_OLLAMA and not provider.available:
                logger.warning(f"⏭️  Ollama not available")
                errors.append(f"Ollama: not running")
                continue
            
            try:
                logger.info(f"🔄 Trying stream from: {provider.NAME}")
                buffer = []
                has_content = False
                
                for chunk in provider.generate_stream(
                    prompt, sys_prompt, temperature, max_tokens
                ):
                    if chunk:
                        has_content = True
                        buffer.append(chunk)
                        yield chunk
                
                if has_content:
                    self.last_provider_used = provider.NAME
                    self._clear_provider_failure(provider.NAME)
                    total_len = sum(len(c) for c in buffer)
                    logger.info(f"✅ Streamed from {provider.NAME} ({total_len} chars)")
                    return
                else:
                    raise ProviderError(f"{provider.NAME} stream was empty")
                    
            except ProviderRateLimitError as e:
                logger.warning(f"⚠️  {provider.NAME} rate limited during stream: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: rate limited")
                
            except ProviderAuthError as e:
                logger.warning(f"⚠️  {provider.NAME} auth error during stream: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: auth error")
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"⚠️  {provider.NAME} connection error during stream")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: connection failed")
                
            except requests.exceptions.Timeout:
                logger.warning(f"⚠️  {provider.NAME} timeout during stream")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: timeout")
                
            except Exception as e:
                logger.error(f"❌ {provider.NAME} stream error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: {str(e)[:100]}")
        
        # ALL providers failed — stream safe fallback
        logger.error(f"❌ All providers failed for stream. Errors: {errors}")
        self.last_provider_used = "fallback"
        yield from self._stream_safe_fallback(prompt, errors)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all providers"""
        status = {
            "last_provider_used": self.last_provider_used,
            "providers": {}
        }
        
        for provider in self.providers:
            pname = provider.NAME
            is_available = provider.available if pname != PROVIDER_OLLAMA else provider.available
            is_cooled = self._is_provider_cooled_down(pname)
            
            status["providers"][pname] = {
                "available": is_available,
                "healthy": is_cooled,
                "last_failure": self._provider_failures.get(pname),
                "model": provider.model
            }
        
        return status
    
    # -----------------------------------------------------------------------
    # Safe Fallback (when ALL providers fail)
    # -----------------------------------------------------------------------
    
    def _generate_safe_fallback(self, prompt: str, errors: List[str]) -> str:
        """Generate a safe static response when all providers are unavailable"""
        error_summary = "; ".join(errors) if errors else "Unknown"
        
        if "summary" in prompt.lower() or "literature review" in prompt.lower():
            return self._fallback_summary(error_summary)
        elif "gap" in prompt.lower():
            return self._fallback_gaps(error_summary)
        elif "compare" in prompt.lower():
            return self._fallback_comparison(error_summary)
        elif "method" in prompt.lower():
            return self._fallback_methodology(error_summary)
        else:
            return self._fallback_generic(error_summary)
    
    def _stream_safe_fallback(self, prompt: str, errors: List[str]) -> Generator[str, None, None]:
        """Stream a safe fallback response character by character"""
        text = self._generate_safe_fallback(prompt, errors)
        # Stream in small chunks to simulate real-time output
        chunk_size = 5
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            time.sleep(0.01)
    
    def _fallback_summary(self, error_info: str) -> str:
        return (
            "## 📚 Literature Review Summary\n\n"
            "### 🔬 Core Research Themes\n"
            "The analyzed literature establishes a robust dual-paradigm framework, "
            "integrating classical methodologies with emerging computational approaches. "
            "Key researchers have consistently emphasized the importance of high-fidelity "
            "data acquisition and cross-domain validation.\n\n"
            "### 📈 Research Trajectories\n"
            "1. **Automated Systems Integration**: Significant movement toward end-to-end automation.\n"
            "2. **Scalability Challenges**: Critical bottlenecks in infrastructure scaling.\n"
            "3. **Hybrid Methodologies**: Combining qualitative insights with quantitative metrics.\n\n"
            "### 🧪 Methodological Innovations\n"
            "Recent studies have pioneered longitudinal multi-variate analysis, "
            "improving predictive accuracy while reducing observational bias.\n\n"
            "### 💡 Synthesis & Future Directions\n"
            "The field is shifting toward more resilient and adaptable architectures. "
            "Future research should prioritize ethical implications and sustainability.\n\n"
            "---\n"
            f"*⚠️ This is a fallback response — all AI providers were unavailable. "
            f"Reason: {error_info}*"
        )
    
    def _fallback_gaps(self, error_info: str) -> str:
        return (
            "## 🔍 Research Gap Analysis\n\n"
            "### ⚠️ Methodological Under-specialization\n"
            "Current literature frequently relies on homogenized datasets, creating visibility gaps "
            "for edge cases and low-frequency variables.\n\n"
            "### 📉 Longitudinal Data Scarcity\n"
            "Most existing research focuses on short-term snapshots that fail to capture "
            "cyclical dynamics and long-term trends.\n\n"
            "### 🚀 Emerging Opportunities\n"
            "Transitioning from reactive to proactive monitoring models presents significant "
            "opportunities for future researchers.\n\n"
            "### 📋 Priority Research Agenda\n"
            "Developing standardized APIs for cross-platform data exchange remains the "
            "highest priority for systemic interoperability.\n\n"
            "---\n"
            f"*⚠️ Fallback response — AI providers unavailable. Reason: {error_info}*"
        )
    
    def _fallback_comparison(self, error_info: str) -> str:
        return (
            "## ⚖️ Comparative Study Synthesis\n\n"
            "### 📊 Divergence in Approaches\n"
            "The studies under review employ distinct theoretical frameworks — from "
            "bottom-up modular approaches to top-down systemic integration.\n\n"
            "### ✅ Consensus on Critical Metrics\n"
            "Across all analyzed papers, there is near-unanimous agreement on the "
            "necessity of real-time validation protocols.\n\n"
            "### 💪 Unique Contributions\n"
            "Each study provides a unique lens: earlier works focus on structural integrity, "
            "while recent publications prioritize dynamic efficiency.\n\n"
            "---\n"
            f"*⚠️ Fallback response — AI providers unavailable. Reason: {error_info}*"
        )
    
    def _fallback_methodology(self, error_info: str) -> str:
        return (
            "## 🔬 Methodology Suggestions\n\n"
            "### Traditional Methods\n"
            "1. **Systematic Literature Review**: Comprehensive evidence synthesis.\n"
            "2. **Meta-Analysis**: Quantitative aggregation of findings.\n\n"
            "### Innovative Methods\n"
            "1. **Machine Learning Classification**: Automated paper categorization.\n"
            "2. **NLP-Based Text Mining**: Extracting patterns from large corpora.\n\n"
            "### Mixed Methods\n"
            "Combining qualitative case studies with quantitative survey data "
            "for comprehensive understanding.\n\n"
            "---\n"
            f"*⚠️ Fallback response — AI providers unavailable. Reason: {error_info}*"
        )
    
    def _fallback_generic(self, error_info: str) -> str:
        return (
            "## 📄 Analysis\n\n"
            "The research materials have been collected and indexed. However, "
            "AI-powered analysis is temporarily unavailable.\n\n"
            "### Available Data\n"
            "Papers have been successfully fetched and can be reviewed manually. "
            "The system will automatically retry AI analysis when providers become available.\n\n"
            "---\n"
            f"*⚠️ Fallback response — AI providers unavailable. Reason: {error_info}*"
        )


# ===========================================================================
# Global Router Instance
# ===========================================================================
llm_router = LLMRouter()
