from novel_translator.ollama_client import OllamaClient


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {"content": '{"translation":"測試"}'},
            "total_duration": 1,
            "load_duration": 0,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 1,
            "eval_count": 5,
            "eval_duration": 1,
        }


class FakeSession:
    def __init__(self):
        self.request = None

    def post(self, url, json, timeout):
        self.request = (url, json, timeout)
        return FakeResponse()


def test_generate_passes_output_limit_and_request_timeout(monkeypatch):
    session = FakeSession()
    client = OllamaClient()
    monkeypatch.setattr(client, "_session", lambda: session)
    client.generate(
        "model",
        "system",
        "prompt",
        options={"num_predict": 512},
        timeout=180,
    )
    _, payload, timeout = session.request
    assert payload["options"] == {
        "temperature": 0.0,
        "num_predict": 512,
    }
    assert timeout == 180
