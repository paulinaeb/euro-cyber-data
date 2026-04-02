"""
Normalize skills for extraction.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


CYBER_EXACT_MAP: Dict[str, str] = {
    "cyber-security": "cybersecurity",
    "cyber security": "cybersecurity",
    "cyber": "cybersecurity",
    "it security": "information security",
    "it security operations": "security operations",
    "security operations center": "security operations center",
    "soc": "security operations center",
    "security information and event management": "siem",
    "security information event management": "siem",
    "cyber threat intelligence": "cti",
    "cyber threat intelligence cti": "cti",
    "identity and access management": "iam",
    "ibm identity and access management": "iam",
    "access management": "identity and access management",
    "general data protection regulation": "gdpr",
    "information security management system": "isms",
    "information security management": "information security management",
    "payment card industry data security standard": "pci dss",
    "internet protocol suite": "tcp/ip",
    "internet protocol": "ip",
    "voice over ip": "voip",
    "dynamic host configuration protocol": "dhcp",
    "domain name system": "dns",
    "virtual private network": "vpn",
    "transport layer security": "tls",
    "application programming interfaces": "api",
    "representational state transfer": "rest",
    "object oriented programming": "oop",
    "continuous integration and continuous delivery": "ci/cd",
    "continuous integration": "ci",
    "large language models": "llm",
    "software development life cycle": "sdlc",
    "infrastructure as code": "iac",
    "hardware security module": "hsm",
    "threat vulnerability management": "vulnerability management",
    "threat & vulnerability management": "vulnerability management",
    "cybersecurity incident response": "incident response",
    "security incident response": "incident response",
    "major incident management": "incident management",
    "endpoint security": "endpoint security",
    "application security assessments": "application security",
    "security architecture design": "security architecture",
    "it risk management": "risk management",
    "governance risk management and compliance": "grc",
    "governance risk compliance": "grc",
    "governance, risk management, and compliance": "grc",
    "data loss prevention": "dlp",
}

PRODUCT_ALIASES: Dict[str, str] = {
    "azure sentinel": "sentinel",
    "microsoft defender": "defender",
    "microsoft entra id": "entra id",
    "google cloud platform": "gcp",
    "amazon web services": "aws",
    "red hat enterprise linux": "rhel",
    "office 365": "microsoft 365",
}

NOISE_SKILL_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b\d+\s+of\s+\d+\s+skill\s+matches\s+your\s+profile\b",
        re.IGNORECASE,
    ),
    re.compile(r"\byou\s+may\s+be\s+a\s+good\s+fit\b", re.IGNORECASE),
]

@dataclass
class NormalizedSkill:
    raw_skill: str
    normalized_skill: str


def ascii_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def basic_clean(text: str) -> str:
    text = ascii_normalize(str(text).strip())
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_parenthetical_acronym(text: str) -> Tuple[str, Optional[str]]:
    match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)\s*", text)
    if not match:
        return text, None

    main, inside = match.group(1).strip(), match.group(2).strip()
    inside_clean = re.sub(r"[^A-Za-z0-9/+.-]", " ", inside).strip()

    if re.fullmatch(r"[A-Za-z0-9/+.-]{2,12}", inside_clean):
        return main, inside_clean

    return main, None


def normalize_token(text: str) -> Tuple[str, Set[str]]:
    aliases: Set[str] = set()
    text = basic_clean(text)
    text, alias = remove_parenthetical_acronym(text)
    if alias:
        aliases.add(alias.lower())

    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9+.# ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"\bit\b", "it", text)
    text = re.sub(r"\bops\b", "operations", text)
    text = re.sub(r"\bsecops\b", "security operations", text)

    if text in PRODUCT_ALIASES:
        text = PRODUCT_ALIASES[text]

    if text in CYBER_EXACT_MAP:
        text = CYBER_EXACT_MAP[text]

    for a in list(aliases):
        if a in {
            "iam",
            "siem",
            "cti",
            "gdpr",
            "isms",
            "pci dss",
            "tls",
            "dns",
            "vpn",
            "api",
            "oop",
            "llm",
            "sdlc",
            "iac",
            "hsm",
            "dhcp",
            "ip",
            "tcp/ip",
            "voip",
        }:
            aliases.add(a)

    return text, aliases


def split_skill_field(value: str) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def normalize_skill(raw_skill: str) -> NormalizedSkill:
    normalized_skill, _aliases = normalize_token(raw_skill)
    return NormalizedSkill(
        raw_skill=raw_skill,
        normalized_skill=normalized_skill,
    )


def is_noise_skill(raw_skill: str) -> bool:
    text = str(raw_skill).strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in NOISE_SKILL_PATTERNS)
