import re

# Standard regex patterns for common PII
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_REGEX = re.compile(r'\+?\b\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.\s]?\d{4}\b')
IPV4_REGEX = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

def redact_pii(text: str) -> str:
    """
    Scans the text and redacts sensitive PII (emails, phone numbers, IPv4 addresses)
    with placeholder tags like [EMAIL], [PHONE], [IP_ADDRESS].
    """
    if not text:
        return text
    
    # Redact Emails
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    # Redact Phone Numbers
    text = PHONE_REGEX.sub("[PHONE]", text)
    # Redact IP Addresses
    text = IPV4_REGEX.sub("[IP_ADDRESS]", text)
    
    return text
