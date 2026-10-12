"""Shared test doubles."""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage

TEST_INGEST_KEY = "test-ingest-key-not-a-secret"
INGEST_HEADERS = {"Authorization": f"Bearer {TEST_INGEST_KEY}"}


class RecordingFakeChatModel(FakeListChatModel):
    """FakeListChatModel that keeps every list of messages it receives."""

    calls: List[List[BaseMessage]] = []

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.calls = []

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    def last_prompt_text(self) -> str:
        """All message contents of the most recent call, joined by newlines."""
        return "\n".join(str(m.content) for m in self.calls[-1])
