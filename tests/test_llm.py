from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agenttool_rl.llm import OpenAICompatibleBackend


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OpenAICompatibleBackendTests(unittest.TestCase):
    @patch("agenttool_rl.llm.urlopen")
    def test_native_tool_call_is_normalized(self, mocked_urlopen) -> None:
        mocked_urlopen.return_value = FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_dependencies",
                                        "arguments": '{"service":"order-api"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        backend = OpenAICompatibleBackend(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen2.5-3B-Instruct",
            api_key="test-key",
        )
        response = backend.complete(
            [{"role": "user", "content": "diagnose"}],
            [{"name": "get_dependencies", "parameters": {"type": "object"}}],
        )

        self.assertEqual(response.tool_name, "get_dependencies")
        self.assertEqual(response.tool_call_id, "call_1")
        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(payload["model"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tool_choice"], "required")


if __name__ == "__main__":
    unittest.main()
