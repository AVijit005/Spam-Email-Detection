from __future__ import annotations

import re

ENGLISH_STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such",
    "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "should",
    "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "weren", "wouldn",
})

DEFAULT_SPAM_THRESHOLD = 0.55
DEFAULT_SUBJECT_WEIGHT = 1

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d[\d\-\(\)\s]{6,}\d\b")
MONEY_PATTERN = re.compile(
    r"[\$£€¥₹]\s*\d[\d,\.\s]*\d|\d[\d,\.\s]*\d\s*[\$£€¥₹]"
    r"|\b\d+(?:[\.,]\d{2})?\s*(?:dollars|pounds|euros|usd|eur|gbp|inr)\b",
    re.IGNORECASE,
)
DOMAIN_PATTERN = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$")
MIXED_TOKEN_PATTERN = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b", re.IGNORECASE)
IP_IN_URL_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
SHORTENED_URL_PATTERN = re.compile(
    r"https?://(?:bit\.ly|t\.co|tinyurl\.com|ow\.ly|goo\.gl|buff\.ly|is\.gd|"
    r"shorte\.st|adf\.ly|bc\.vc|rebrand\.ly|cutt\.ly)/\S+"
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_COMMENT_PATTERN = re.compile(r"<!--[\s\S]*?-->")
CSS_HIDDEN_PATTERN = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0|"
    r"color\s*:\s*(?:white|#fff|#ffffff|transparent)\s*(?:on|with)?\s*(?:white|#fff|#ffffff))",
    re.IGNORECASE,
)
UNICODE_OBFUSCATION_PATTERN = re.compile(r"[^\x00-\x7F]+")
HOMOGRAPH_CHAR_PATTERN = re.compile(r"[ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝàáâãäåçèéêëìíîïðñòóôõöøùúûüýÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ]")

URGENCY_KEYWORDS = {
    "urgent", "immediately", "asap", "suspended", "expire", "expired",
    "deadline", "warning", "alert", "critical", "important", "attention",
    "final notice", "last chance", "limited time", "act now", "respond now",
    "before it's too late", "don't miss", "hurry", "time sensitive",
}

ACCOUNT_KEYWORDS = {
    "account", "password", "login", "signin", "security", "verify",
    "verification", "identity", "otp", "bank", "credential", "pin",
    "ssn", "social security", "date of birth", "mother's maiden",
    "unlock", "deactivated", "billing", "invoice", "statement",
    "payment method", "credit card", "debit card", "cvv", "expiry",
}

CALL_TO_ACTION_KEYWORDS = {
    "click", "claim", "confirm", "reset", "verify", "open", "download",
    "visit", "login", "sign in", "sign up", "register", "subscribe",
    "unsubscribe", "accept", "agree", "enable",
    "allow", "grant", "authorize", "proceed", "continue",
}

PROMOTIONAL_KEYWORDS = {
    "offer", "discount", "sale", "coupon", "deal", "shop", "weekend",
    "save", "percent", "shipping", "clearance", "bogo", "promo",
    "voucher", "gift card", "black friday", "cyber monday",
}

CONVERSATIONAL_KEYWORDS = {
    "lunch", "coffee", "dinner", "meeting", "office", "today",
    "tomorrow", "plans", "near", "still", "thanks", "regards",
    "cheers", "talk soon", "catch up", "how are you",
}

BUSINESS_KEYWORDS = {
    "review", "report", "slides", "project", "team", "agenda",
    "meeting", "office", "update", "client", "schedule", "deadline",
    "quarter", "budget", "proposal", "contract",
}

FINANCIAL_PHISHING_PHRASES = [
    "tax refund", "unclaimed funds", "inheritance", "lottery winner",
    "you have won", "you've been selected", "claim your prize",
    "claim now", "winner", "won a lottery", "lottery prize",
    "free money", "million dollars", "million pound",
    "wire transfer", "western union", "moneygram",
    "bitcoin", "cryptocurrency", "investment opportunity",
    "double your money", "guaranteed return", "no risk",
    "offshore account", "inheritance claim", "dormant account",
    "unclaimed property", "prize award", "cash prize",
]

TECH_SUPPORT_PHISHING_PHRASES = [
    "your computer is infected", "virus detected", "windows support",
    "microsoft support", "apple support", "tech support",
    "remote access", "teamviewer", "anydesk", "logmein",
    "your ip address", "your router", "your firewall",
    "suspicious activity", "unauthorized access", "security breach",
    "install this software", "run this program",
]

HR_PAYROLL_PHISHING_PHRASES = [
    "update your direct deposit", "salary revision", "hr verification",
    "payroll update", "w2 form", "tax form", "employee portal",
    "benefits enrollment", "open enrollment", "401k update",
    "change your password", "password will expire",
    "confirm your identity", "identity verification",
]

SHIPPING_PHISHING_PHRASES = [
    "parcel held", "customs fee", "delivery attempt failed",
    "package undelivered", "reschedule delivery", "track your package",
    "shipping confirmation", "order confirmation", "payment required",
    "additional postage", "address verification needed",
]

SOCIAL_ENGINEERING_PHRASES = [
    "dear lucky winner", "dear beneficiary", "dear friend",
    "i need your help", "confidential", "for your eyes only",
    "do not tell anyone", "keep this secret", "trust me",
    "i am a prince", "foreign dignitary", "business proposal",
    "confidential business", "mutual benefit", "kindly",
    "greetings to you", "with due respect",
]

CREDENTIAL_HARVESTING_PHRASES = [
    "click here to verify", "verify your account immediately",
    "account suspended", "account has been suspended",
    "account will be closed", "account deactivated",
    "login attempt blocked", "unusual login", "new sign-in",
    "confirm your email", "validate your account",
    "update your information", "billing information",
    "payment declined", "card expired", "update payment",
]

PHISHING_PHRASES = (
    FINANCIAL_PHISHING_PHRASES
    + TECH_SUPPORT_PHISHING_PHRASES
    + HR_PAYROLL_PHISHING_PHRASES
    + SHIPPING_PHISHING_PHRASES
    + SOCIAL_ENGINEERING_PHRASES
    + CREDENTIAL_HARVESTING_PHRASES
)

SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".pw",
    ".cc", ".ws", ".bid", ".trade", ".webcam", ".science",
    ".party", ".date", ".download", ".loan", ".racing", ".review",
    ".country", ".faith", ".cricket", ".men", ".win", ".stream",
}

HIGH_RISK_TLDS = {
    ".zip", ".mov", ".nexe", ".vbs", ".exe", ".scr",
}

ATTACHMENT_PHRASES = [
    "invoice", "receipt", "statement", "purchase order",
    "wire confirmation", "payment confirmation", "voicemail",
    "fax", "scanned document", "document shared", "secure message",
    "encrypted message", "view attachment", "open attachment",
    "download file", "attached file", "see attached",
]

ATTACHMENT_EXTENSIONS = {
    ".pdf", ".zip", ".rar", ".7z", ".doc", ".docm", ".docx",
    ".xls", ".xlsm", ".xlsx", ".ppt", ".pptm", ".pptx",
    ".iso", ".img", ".js", ".vbs", ".ps1", ".bat", ".cmd",
    ".scr", ".exe", ".msi", ".hta", ".jar",
}

META_FEATURE_NAMES = [
    "url_count",
    "caps_ratio",
    "exclamation_count",
    "question_count",
    "money_count",
    "phone_count",
    "word_count",
    "avg_word_length",
    "digit_ratio",
    "spam_phrase_hits",
    "urgency_hits",
    "account_hits",
    "call_to_action_hits",
    "symbol_ratio",
    "percent_hits",
    "mixed_token_hits",
    "unique_url_domains",
    "shortened_url_count",
    "ip_url_count",
    "suspicious_tld_count",
    "high_risk_tld_count",
    "url_to_text_ratio",
    "attachment_indicators",
    "html_tag_count",
    "hidden_element_indicators",
    "homograph_char_ratio",
    "unicode_obfuscation_ratio",
    "flesch_reading_ease",
    "type_token_ratio",
    "imperative_verb_ratio",
    "promotional_hits",
    "credential_harvesting_hits",
]

META_FEATURE_LABELS = {
    "url_count": "contains links",
    "caps_ratio": "uses uppercase emphasis",
    "exclamation_count": "uses repeated exclamation marks",
    "question_count": "uses multiple question marks",
    "money_count": "mentions money values",
    "phone_count": "contains a phone number",
    "word_count": "message length",
    "avg_word_length": "long token pattern",
    "digit_ratio": "contains many digits",
    "spam_phrase_hits": "matches phishing phrases",
    "urgency_hits": "contains urgency language",
    "account_hits": "contains account-security language",
    "call_to_action_hits": "contains calls to action",
    "symbol_ratio": "contains many symbols",
    "percent_hits": "contains discount-style percentages",
    "mixed_token_hits": "contains mixed letter-number tokens",
    "unique_url_domains": "links to multiple domains",
    "shortened_url_count": "uses shortened URLs",
    "ip_url_count": "links to IP addresses",
    "suspicious_tld_count": "links to suspicious TLDs",
    "high_risk_tld_count": "links to high-risk file TLDs",
    "url_to_text_ratio": "high link density",
    "attachment_indicators": "mentions attachments",
    "html_tag_count": "contains HTML markup",
    "hidden_element_indicators": "contains hidden elements",
    "homograph_char_ratio": "uses lookalike characters",
    "unicode_obfuscation_ratio": "uses Unicode obfuscation",
    "flesch_reading_ease": "text readability score",
    "type_token_ratio": "vocabulary richness",
    "imperative_verb_ratio": "uses command language",
    "promotional_hits": "contains promotional language",
    "credential_harvesting_hits": "matches credential harvesting patterns",
}
