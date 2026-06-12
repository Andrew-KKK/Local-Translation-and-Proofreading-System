from dataclasses import dataclass
import requests


class OllamaError(RuntimeError):
    pass


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    timeout: int = 600

    def _session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session

    def list_models(self) -> list[str]:
        try:
            response = self._session().get(
                f"{self.base_url.rstrip('/')}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            return [item["name"] for item in response.json().get("models", [])]
        except requests.RequestException as exc:
            raise OllamaError("無法取得 Ollama 模型清單。") from exc

    def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        format_schema: dict | None = None,
    ) -> str:
        try:
            payload = {
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.0},
            }
            if format_schema:
                payload["format"] = format_schema
            response = self._session().post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.ConnectionError as exc:
            raise OllamaError("無法連接 Ollama，請先啟動 Ollama 服務。") from exc
        except requests.Timeout as exc:
            raise OllamaError("Ollama 回應逾時，請縮短文本後重試。") from exc
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise OllamaError(f"Ollama 請求失敗：{detail}") from exc
        except (KeyError, ValueError) as exc:
            raise OllamaError("Ollama 回傳了無法解析的資料。") from exc
