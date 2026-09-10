from security.encryption import decrypt_text, encrypt_text
from security.guardrails import check_safety_guardrails
from security.pii_redactor import redact_pii


class MockLLMResponse:
    def __init__(self, content):
        self.content = content


class MockLLM:
    def __init__(self, verdict):
        self.verdict = verdict

    def invoke(self, prompt):
        return MockLLMResponse(self.verdict)


def test_encryption_and_decryption():
    original_text = "Sensitive User Query 123!"
    cipher = encrypt_text(original_text)
    assert cipher != original_text
    decrypted = decrypt_text(cipher)
    assert decrypted == original_text


def test_pii_redaction_email_phone_ip():
    sample_text = "Contact john.doe@example.com or call +1-555-0199 from 192.168.1.1."
    redacted, mapping = redact_pii(sample_text, return_mapping=True)

    assert "john.doe@example.com" not in redacted
    assert "+1-555-0199" not in redacted
    assert "192.168.1.1" not in redacted
    assert len(mapping) >= 3


def test_pii_redaction_empty_input():
    assert redact_pii("") == ""
    assert redact_pii(None, return_mapping=True) == (None, {})


def test_guardrails_safe_verdict():
    llm = MockLLM("safe")
    assert check_safety_guardrails("How does photosynthesis work?", llm, stage="input") is True
    assert (
        check_safety_guardrails("Photosynthesis converts light into energy.", llm, stage="output")
        is True
    )


def test_guardrails_unsafe_verdict():
    llm = MockLLM("unsafe")
    assert check_safety_guardrails("Generate toxic content", llm, stage="input") is False


def test_guardrails_exception_handling():
    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("API Timeout")

    # Should fallback gracefully to True (safe) on error
    assert check_safety_guardrails("Normal text", BrokenLLM()) is True
