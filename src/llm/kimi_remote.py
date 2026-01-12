import os
import time
from typing import Optional
from openai import OpenAI

DEFAULT_MODEL = "moonshotai/kimi-k2-instruct-0905"
# export NVAPI_KEY="Tự lấy đi nhé hehe"


class KimiGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 30,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv("NVAPI_KEY")
        if not self.api_key:
            raise RuntimeError("NVAPI_KEY not set")

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=timeout,
        )
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        last_err = None

        for _ in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=0.9,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()

            except Exception as e:
                last_err = e
                time.sleep(1)

        raise RuntimeError(f"Kimi generation failed: {last_err}")
