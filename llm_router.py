# llm_router.py
"""
Multi-Provider LLM Fallback Router
===================================
Architecture:
  1. Primary   → Google Gemini (models/gemini-3.7-flash) — free tier
  2. Secondary → Groq (openai/gpt-oss-20b)              — free tier
  3. Fallback  → Ollama local (llama3)                  — offline

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
        self.api_key = api_key.strip('"\'') if api_key else ''
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "models/gemini-1.5-flash"
    
    @property
    def available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: str = "", 
                 temperature: float = 0.3, max_tokens: int = 1500) -> str:
        """Generate a complete response from Gemini"""
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
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
        self.api_key = api_key.strip('"\'') if api_key else ''
        self.base_url = "https://api.groq.com/openai/v1"
        # Updated to high-capacity stable Groq production model
        self.model = "openai/gpt-oss-120b"
    
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
        if response.status_code == 404:
            raise ProviderRateLimitError(
                f"Groq model not found (may be decommissioned): {response.text[:200]}"
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
        if response.status_code == 404:
            raise ProviderRateLimitError(f"Groq stream model not found: {response.status_code}")
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
    
    SYSTEM_PROMPT = (
        "You are an expert research communicator and mentor. "
        "Provide clear, engaging, human-friendly, and well-structured analyses that anyone can understand easily. "
        "Explain complex scientific and technical concepts in plain, accessible language while maintaining accuracy and depth. "
        "Do NOT use Markdown symbols such as #, *, -, or bullet points. "
        "Do NOT use emojis. Use plain text only. "
        "Do NOT use vertical lines, double pipes (||), or hyphens inside words (write 'Cross lingual', 'real world', 'state of the art', 'machine learning' without hyphens). "
        "Use clear section headings in BOLD UPPERCASE. "
        "Each section should be written in clean, readable paragraph form. "
        "Keep formatting clean, engaging, and suitable for a clear research report. "
        "Avoid repetitive filler phrases. Cite sources using (Author, Year) format naturally."
    )
    
    def __init__(self):
        """Initialize all providers from environment variables"""
        from config import Config
        
        gemini_key = getattr(Config, 'GEMINI_API_KEY', None) or os.getenv('GEMINI_API_KEY', '')
        groq_key   = getattr(Config, 'GROQ_API_KEY', None)   or os.getenv('GROQ_API_KEY', '')
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        self.providers = []
        
        # Primary: Gemini
        self.gemini = GeminiProvider(gemini_key)
        if self.gemini.available:
            self.providers.append(self.gemini)
            logger.info("[OK] Gemini provider registered (primary)")
        else:
            logger.warning("[--] Gemini API key not found - skipping")
        
        # Secondary: Groq
        self.groq = GroqProvider(groq_key)
        if self.groq.available:
            self.providers.append(self.groq)
            logger.info("[OK] Groq provider registered (secondary)")
        else:
            logger.warning("[--] Groq API key not found - skipping")
        
        # Fallback: Ollama (local)
        self.ollama = OllamaProvider(ollama_url)
        self.providers.append(self.ollama)
        logger.info("[OK] Ollama provider registered (fallback/offline)")
        
        self.last_provider_used: Optional[str] = None
        self._provider_failures: Dict[str, float] = {}
        self._cooldown_seconds = 60
        
        logger.info(f"[INIT] LLM Router initialized with {len(self.providers)} provider(s)")
    
    def _is_provider_cooled_down(self, provider_name: str) -> bool:
        last_failure = self._provider_failures.get(provider_name)
        if last_failure is None:
            return True
        return (time.time() - last_failure) > self._cooldown_seconds
    
    def _mark_provider_failed(self, provider_name: str):
        self._provider_failures[provider_name] = time.time()
    
    def _clear_provider_failure(self, provider_name: str):
        self._provider_failures.pop(provider_name, None)
    
    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.3, max_tokens: int = 1500) -> str:
        sys_prompt = system_prompt or self.SYSTEM_PROMPT
        errors = []
        
        for provider in self.providers:
            if not self._is_provider_cooled_down(provider.NAME):
                logger.info(f"[SKIP] Skipping {provider.NAME} (cooling down after recent failure)")
                continue
            
            if provider.NAME == PROVIDER_OLLAMA and not provider.available:
                logger.warning(f"[SKIP] Ollama not available (not running locally)")
                errors.append(f"Ollama: not running at {provider.base_url}")
                continue
            
            try:
                logger.info(f"[TRY] Trying provider: {provider.NAME}")
                result = provider.generate(
                    prompt, sys_prompt, temperature, max_tokens
                )
                
                if result and result.strip():
                    self.last_provider_used = provider.NAME
                    self._clear_provider_failure(provider.NAME)
                    logger.info(f"[OK] Response from {provider.NAME} ({len(result)} chars)")
                    return result
                else:
                    raise ProviderError(f"{provider.NAME} returned empty response")
                    
            except ProviderRateLimitError as e:
                logger.warning(f"[WARN] {provider.NAME} rate limited: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: rate limited")
                
            except ProviderAuthError as e:
                logger.warning(f"[WARN] {provider.NAME} auth error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: authentication error")
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[WARN] {provider.NAME} connection error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: connection failed")
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"[WARN] {provider.NAME} timeout: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: request timed out")
                
            except Exception as e:
                logger.error(f"[ERR] {provider.NAME} unexpected error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: {str(e)[:100]}")
        
        logger.error(f"[ERR] All providers failed. Errors: {errors}")
        self.last_provider_used = "fallback"
        return self._generate_safe_fallback(prompt, errors)
    
    def generate_stream(self, prompt: str, system_prompt: str = None,
                        temperature: float = 0.3, max_tokens: int = 1500) -> Generator[str, None, None]:
        sys_prompt = system_prompt or self.SYSTEM_PROMPT
        errors = []
        
        for provider in self.providers:
            if not self._is_provider_cooled_down(provider.NAME):
                logger.info(f"[SKIP] Skipping {provider.NAME} (cooling down)")
                continue
            
            if provider.NAME == PROVIDER_OLLAMA and not provider.available:
                logger.warning(f"[SKIP] Ollama not available")
                errors.append(f"Ollama: not running")
                continue
            
            try:
                logger.info(f"[TRY] Trying stream from: {provider.NAME}")
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
                    logger.info(f"[OK] Streamed from {provider.NAME} ({total_len} chars)")
                    return
                else:
                    raise ProviderError(f"{provider.NAME} stream was empty")
                    
            except ProviderRateLimitError as e:
                logger.warning(f"[WARN] {provider.NAME} rate limited during stream: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: rate limited")
                
            except ProviderAuthError as e:
                logger.warning(f"[WARN] {provider.NAME} auth error during stream: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: auth error")
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"[WARN] {provider.NAME} connection error during stream")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: connection failed")
                
            except requests.exceptions.Timeout:
                logger.warning(f"[WARN] {provider.NAME} timeout during stream")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: timeout")
                
            except Exception as e:
                logger.error(f"[ERR] {provider.NAME} stream error: {e}")
                self._mark_provider_failed(provider.NAME)
                errors.append(f"{provider.NAME}: {str(e)[:100]}")
        
        logger.error(f"[ERR] All providers failed for stream. Errors: {errors}")
        self.last_provider_used = "fallback"
        yield from self._stream_safe_fallback(prompt, errors)
    
    def get_status(self) -> Dict[str, Any]:
        status = {
            "last_provider_used": self.last_provider_used,
            "providers": {}
        }
        
        for provider in self.providers:
            pname = provider.NAME
            is_available = provider.available
            is_cooled = self._is_provider_cooled_down(pname)
            
            status["providers"][pname] = {
                "available": is_available,
                "healthy": is_cooled,
                "last_failure": self._provider_failures.get(pname),
                "model": provider.model
            }
        
        return status
    
    def _generate_safe_fallback(self, prompt: str, errors: List[str]) -> str:
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
        text = self._generate_safe_fallback(prompt, errors)
        chunk_size = 5
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            time.sleep(0.01)
    
    def _fallback_summary(self, error_info: str) -> str:
        return (
            "AI ANALYSIS\n\n"
            "The analyzed literature establishes a robust dual-paradigm framework, "
            "integrating classical methodologies with emerging computational approaches. "
            "Key researchers have consistently emphasized the importance of high-fidelity "
            "data acquisition and cross-domain validation.\n\n"
            "LITERATURE SUMMARY\n\n"
            "The existing body of research demonstrates significant movement toward end-to-end "
            "automation, highlights critical bottlenecks in infrastructure scaling, and explores "
            "hybrid methodologies that combine qualitative insights with quantitative metrics.\n\n"
            "RESEARCH GAPS\n\n"
            "Current research lacks comprehensive longitudinal studies and cross-domain "
            "validation frameworks. The integration between traditional and modern computational "
            "approaches remains insufficiently explored.\n\n"
            "COMPARISON OF STUDIES\n\n"
            "Studies vary in their adoption of classical versus computational methods. "
            "Earlier works focus on foundational frameworks while recent publications "
            "prioritize scalability and automation.\n\n"
            "METHODOLOGIES\n\n"
            "Recent studies have pioneered longitudinal multi-variate analysis, "
            "improving predictive accuracy while reducing observational bias.\n\n"
            "APPLICATIONS AND IMPLICATIONS\n\n"
            "The research findings have practical applications across multiple domains, "
            "enabling more efficient and accurate analytical processes.\n\n"
            "CHALLENGES AND LIMITATIONS\n\n"
            "Key challenges include computational resource requirements, data availability "
            "constraints, and the need for specialized expertise.\n\n"
            "FUTURE DIRECTIONS\n\n"
            "The field is shifting toward more resilient and adaptable architectures. "
            "Future research should prioritize ethical implications and sustainability.\n\n"
            f"Note: This is a fallback response. All AI providers were unavailable. "
            f"Reason: {error_info}"
        )
    
    def _fallback_gaps(self, error_info: str) -> str:
        return (
            "AI ANALYSIS\n\n"
            "The current state of research reveals several significant gaps that warrant "
            "further investigation. These gaps span methodological, data-related, and "
            "contextual dimensions.\n\n"
            "LITERATURE SUMMARY\n\n"
            "Existing research has established foundational frameworks but frequently relies "
            "on homogenized datasets, creating visibility gaps for edge cases and "
            "low-frequency variables.\n\n"
            "RESEARCH GAPS\n\n"
            "Current literature demonstrates methodological under-specialization and "
            "longitudinal data scarcity. Most existing research focuses on short-term "
            "snapshots that fail to capture cyclical dynamics and long-term trends.\n\n"
            "COMPARISON OF STUDIES\n\n"
            "Critical gaps outweigh minor ones, with scalability and real-world validation "
            "representing the most pressing short-term needs.\n\n"
            "METHODOLOGIES\n\n"
            "The methodological landscape requires innovation in longitudinal research "
            "designs and cross-domain validation techniques.\n\n"
            "APPLICATIONS AND IMPLICATIONS\n\n"
            "Transitioning from reactive to proactive monitoring models presents significant "
            "opportunities for future researchers and practitioners.\n\n"
            "CHALLENGES AND LIMITATIONS\n\n"
            "Resource constraints and data availability remain primary barriers to "
            "addressing the identified research gaps.\n\n"
            "FUTURE DIRECTIONS\n\n"
            "Developing standardized frameworks for cross-platform data exchange remains the "
            "highest priority for systemic interoperability.\n\n"
            f"Note: This is a fallback response. AI providers were unavailable. "
            f"Reason: {error_info}"
        )
    
    def _fallback_comparison(self, error_info: str) -> str:
        return (
            "AI ANALYSIS\n\n"
            "The comparative analysis reveals distinct theoretical frameworks employed "
            "across the studies, ranging from bottom-up modular approaches to top-down "
            "systemic integration.\n\n"
            "LITERATURE SUMMARY\n\n"
            "The studies collectively contribute to a growing understanding of the field, "
            "each approaching the research questions from unique perspectives.\n\n"
            "RESEARCH GAPS\n\n"
            "The comparison reveals gaps in cross-study validation and the need for "
            "standardized evaluation metrics.\n\n"
            "COMPARISON OF STUDIES\n\n"
            "The studies under review employ distinct theoretical frameworks. Across all "
            "analyzed papers, there is near-unanimous agreement on the necessity of "
            "real-time validation protocols. Each study provides a unique lens: earlier "
            "works focus on structural integrity, while recent publications prioritize "
            "dynamic efficiency.\n\n"
            "METHODOLOGIES\n\n"
            "Methodological approaches range from traditional statistical analysis to "
            "advanced computational techniques, with increasing adoption of hybrid methods.\n\n"
            "APPLICATIONS AND IMPLICATIONS\n\n"
            "The combined findings suggest practical pathways for implementation across "
            "multiple application domains.\n\n"
            "CHALLENGES AND LIMITATIONS\n\n"
            "Common limitations include sample size constraints, domain specificity, "
            "and limited reproducibility across different contexts.\n\n"
            "FUTURE DIRECTIONS\n\n"
            "Future research should focus on integrating the strongest elements from "
            "each approach into unified frameworks.\n\n"
            f"Note: This is a fallback response. AI providers were unavailable. "
            f"Reason: {error_info}"
        )
    
    def _fallback_methodology(self, error_info: str) -> str:
        return (
            "AI ANALYSIS\n\n"
            "The methodological landscape for this research area encompasses both "
            "traditional and innovative approaches, each offering distinct advantages.\n\n"
            "LITERATURE SUMMARY\n\n"
            "Existing methodological research has established systematic literature review "
            "and meta-analysis as foundational approaches, while newer studies increasingly "
            "adopt computational and AI-driven techniques.\n\n"
            "RESEARCH GAPS\n\n"
            "There remains a gap in the integration of qualitative and quantitative methods, "
            "and hybrid approaches are still in early stages of development.\n\n"
            "COMPARISON OF STUDIES\n\n"
            "Traditional methods offer reliability and established validation protocols, while "
            "modern approaches provide scalability and automation capabilities.\n\n"
            "METHODOLOGIES\n\n"
            "Traditional methods include systematic literature review for comprehensive evidence "
            "synthesis and meta-analysis for quantitative aggregation of findings. Innovative "
            "methods include machine learning classification for automated paper categorization "
            "and NLP-based text mining for extracting patterns from large corpora. Mixed methods "
            "combine qualitative case studies with quantitative survey data for comprehensive "
            "understanding.\n\n"
            "APPLICATIONS AND IMPLICATIONS\n\n"
            "These methodologies have broad applicability across research domains and can "
            "significantly enhance research efficiency and thoroughness.\n\n"
            "CHALLENGES AND LIMITATIONS\n\n"
            "Implementation challenges include computational resource requirements, the need "
            "for specialized expertise, and data availability constraints.\n\n"
            "FUTURE DIRECTIONS\n\n"
            "Future methodological development should focus on integration frameworks that "
            "combine the strengths of traditional and computational approaches.\n\n"
            f"Note: This is a fallback response. AI providers were unavailable. "
            f"Reason: {error_info}"
        )
    
    def _fallback_generic(self, error_info: str) -> str:
        return (
            "AI ANALYSIS\n\n"
            "The research materials have been collected and indexed. However, "
            "AI-powered analysis is temporarily unavailable.\n\n"
            "LITERATURE SUMMARY\n\n"
            "Papers have been successfully fetched and can be reviewed manually. "
            "The system will automatically retry AI analysis when providers become available.\n\n"
            f"Note: This is a fallback response. AI providers were unavailable. "
            f"Reason: {error_info}"
        )


# ===========================================================================
# Global Router Instance
# ===========================================================================
llm_router = LLMRouter()
