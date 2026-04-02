import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from agents.summarizer import SummarizerAgent
from config.schemas import IncidentReport
from config.constants import DisasterType, Severity


def _make_incident(title="Flood in Chennai", d_type="flood", sev="critical") -> IncidentReport:
    """Helper: build a minimal IncidentReport for tests."""
    return IncidentReport(
        id="TEST001", title=title, source="mock",
        location="Chennai, India", lat=13.08, lon=80.27,
        disaster_type=d_type, severity=sev,
        timestamp=datetime.now(timezone.utc),
        raw_text="Severe flooding in Adyar basin. 50,000 displaced. NDRF deployed.",
    )

def _mock_llm_response(data: dict) -> str:
    """Return a properly formatted JSON string as the mocked LLM output."""
    return json.dumps(data)


class TestSummarizerAgent:

    @patch("agents.base.OpenAI")
    def test_returns_summary_result_on_valid_input(self, MockOpenAI):
        """Happy path: valid input → SummaryResult with all fields."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = _mock_llm_response({
            "summary": "Flooding in Chennai affects 50,000 residents.",
            "key_facts": ["50,000 displaced", "NDRF deployed", "Adyar river overflow"],
            "immediate_risk": "critical",
            "location_refined": "Adyar basin, Chennai",
            "casualties_mentioned": None,
            "immediate_action": "Evacuate residents in Kotturpuram immediately.",
        })
        mock_resp.usage.total_tokens = 150
        MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

        agent = SummarizerAgent()
        result = agent.summarize(_make_incident())

        assert result is not None
        assert "50,000" in result.summary
        assert len(result.key_facts) == 3
        assert result.immediate_risk == Severity.CRITICAL

    @patch("agents.base.OpenAI")
    def test_returns_none_on_empty_llm_response(self, MockOpenAI):
        """Empty LLM response → returns None gracefully, no crash."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = ""
        mock_resp.usage.total_tokens = 0
        MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

        agent = SummarizerAgent()
        result = agent.summarize(_make_incident())
        assert result is None

    @patch("agents.base.OpenAI")
    def test_strips_markdown_fences_from_llm_output(self, MockOpenAI):
        """LLM sometimes returns ```json ... ``` — must be stripped."""
        raw_with_fences = """```json
{"summary": "Clean summary.", "key_facts": ["fact1"], "immediate_risk": "high",
 "location_refined": null, "casualties_mentioned": null, "immediate_action": "Act now."}
```"""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = raw_with_fences
        mock_resp.usage.total_tokens = 80
        MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

        agent = SummarizerAgent()
        result = agent.summarize(_make_incident())
        assert result is not None
        assert result.summary == "Clean summary."

    @patch("agents.base.OpenAI")
    def test_maps_risk_level_correctly(self, MockOpenAI):
        """immediate_risk string from LLM must map to correct Severity enum."""
        for risk_str, expected_sev in [
            ("low", Severity.LOW), ("medium", Severity.MEDIUM),
            ("high", Severity.HIGH), ("critical", Severity.CRITICAL),
        ]:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = _mock_llm_response({
                "summary": "Test.", "key_facts": [], "immediate_risk": risk_str,
                "location_refined": None, "casualties_mentioned": None,
                "immediate_action": "Act.",
            })
            mock_resp.usage.total_tokens = 50
            MockOpenAI.return_value.chat.completions.create.return_value = mock_resp
            agent = SummarizerAgent()
            result = agent.summarize(_make_incident())
            assert result.immediate_risk == expected_sev, f"Failed for {risk_str}"

    @patch("agents.base.OpenAI")
    def test_handles_malformed_json_gracefully(self, MockOpenAI):
        """If LLM returns non-JSON, must return None without raising."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "This is not JSON at all, just text."
        mock_resp.usage.total_tokens = 10
        MockOpenAI.return_value.chat.completions.create.return_value = mock_resp

        agent = SummarizerAgent()
        result = agent.summarize(_make_incident())
        assert result is None