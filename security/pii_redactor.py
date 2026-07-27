import re

# Standard regex patterns for common PII
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_REGEX = re.compile(r'\+?\b\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.\s]?\d{4}\b')
IPV4_REGEX = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

def redact_pii(text: str, return_mapping: bool = False):
    """
    Scans the text and redacts sensitive PII (emails, phone numbers, IPv4 addresses)
    with placeholder tags like redacted_email, redacted_phone, redacted_ip.
    """
    if not text:
        return (text, {}) if return_mapping else text
    
    mapping = {}
    
    # Redact Emails
    emails = EMAIL_REGEX.findall(text)
    for idx, email in enumerate(emails):
        placeholder = f"redacted_email_{idx}"
        mapping[placeholder] = email
        text = text.replace(email, placeholder)
        
    # Redact Phone Numbers
    phones = PHONE_REGEX.findall(text)
    for idx, phone in enumerate(phones):
        placeholder = f"redacted_phone_{idx}"
        mapping[placeholder] = phone
        text = text.replace(phone, placeholder)
        
    # Redact IP Addresses
    ips = IPV4_REGEX.findall(text)
    for idx, ip in enumerate(ips):
        placeholder = f"redacted_ip_{idx}"
        mapping[placeholder] = ip
        text = text.replace(ip, placeholder)
        
    # Standard fallback tags for general regex scrubbing (in case elements missed by findall)
    text = EMAIL_REGEX.sub("redacted_email", text)
    text = PHONE_REGEX.sub("redacted_phone", text)
    text = IPV4_REGEX.sub("redacted_ip", text)
    
    return (text, mapping) if return_mapping else text
