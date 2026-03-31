"""
Data Preprocessing Script
Cleans, transforms, and prepares raw data for the pipeline.
Saves preprocessed data to data/preprocessed/

"""

import json
import html
import re
import pandas as pd
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import RAW_DATA_DIR, PREPROCESSED_DIR
from src.utils.cli_args import add_sample_mode_arguments, is_valid_sample_size
from src.utils.markup_detection import (
    COLLAPSE_NEWLINES_PATTERN,
    DECORATIVE_SEPARATOR_PATTERN,
    find_records_with_markup,
    GENDER_MARKER_PATTERN,
    get_detected_markup_types,
    HTML_TAG_PATTERN,
    INLINE_CODE_PATTERN,
    MARKDOWN_BOLD_PATTERN,
    MARKDOWN_LINK_PATTERN,
    MARKDOWN_UNDERSCORE_PATTERN,
    NEWLINE_SPACING_PATTERN,
    NEWLINE_TO_PERIOD_PATTERN,
    remove_asterisk_clusters,
    remove_broken_or_partial_urls,
    remove_email_addresses,
    remove_empty_wrappers,
    remove_gender_marker_tokens,
    remove_phone_numbers,
    remove_redacted_placeholders,
    QUOTE_CHARACTER_PATTERN,
    RAW_URL_PATTERN,
    remove_emoji_like_unicode,
    WHITESPACE_PATTERN,
)
from src.utils.sampling import sample_collection
from src.preprocessing.invalid_record_detection import (
    get_all_critical_fields_invalid_mask,
)
from src.preprocessing.language_detection import detect_language_distribution, invalid_content_mask


JOB_POSTINGS_COLUMNS_TO_DROP = [
    'Insight',
    'Job State',
    'Detail URL',
    'Company Name',
    'Company Description',
    'Company Website',
    'Company Logo',
    'Company Apply Url',
    'Employee Count',
    'Headquarters',
    'Company Founded',
    'Specialties',
    'Hiring Manager Title',
    'Hiring Manager Subtitle',
    'Hiring Manager Title Insight',
    'Hiring Manager Profile',
    'Hiring Manager Image',
    'Poster Id',
    'Industry',
]

LOCATION_TO_COUNTRY_OVERRIDES = {
    "Warsaw Metropolitan Area": "Poland",
    "Cork Metropolitan Area": "Ireland",
    "Greater Stockholm Metropolitan Area": "Sweden",
    "Bucharest Metropolitan Area": "Romania",
    "Brussels Metropolitan Area": "Belgium",
    "Gdansk Metropolitan Area": "Poland",
    "Rotterdam and The Hague": "Netherlands",
    "Greater Marseille Metropolitan Area": "France",
    "Amsterdam Area": "Netherlands",
    "Greater Groningen Area": "Netherlands",
    "Tampere Metropolitan Area": "Finland",
    "Stuttgart Region": "Germany",
    "Porto Metropolitan Area": "Portugal",
    "Greater Milan Metropolitan Area": "Italy",
    "Greater Madrid Metropolitan Area": "Spain",
    "Greater Sevilla Metropolitan Area": "Spain",
    "Arnhem-Nijmegen Region": "Netherlands",
    "Lisbon Metropolitan Area": "Portugal",
    "Greater Barcelona Metropolitan Area": "Spain",
    "Greater Zaragoza Metropolitan Area": "Spain",
    "Greater Hradec Kralove Area": "Czechia",
    "Copenhagen Metropolitan Area": "Denmark",
    "Greater Paris Metropolitan Region": "France",
    "Cologne Bonn Region": "Germany",
    "England Metropolitan Area": "United Kingdom",
    "Brno Metropolitan Area": "Czechia",
    "Plzen Metropolitan Area": "Czechia",
    "Prague Metropolitan Area": "Czechia",
    "Sofia Metropolitan Area": "Bulgaria",
    "Plovdiv Metropolitan Area": "Bulgaria",
    "Athens Metropolitan Area": "Greece",
    "Thessaloniki Metropolitan Area": "Greece",
    "Geneva Metropolitan Area": "Switzerland",
    "Basel Metropolitan Area": "Switzerland",
    "Zürich Metropolitan Area": "Switzerland",
    "Lugano Metropolitan Area": "Switzerland",
    "Lausanne Metropolitan Area": "Switzerland",
    "Lucerne Metropolitan Area": "Switzerland",
    "Sankt Gallen Metropolitan Area": "Switzerland",
    "Greater Turin Metropolitan Area": "Italy",
    "Italy Metropolitan Area": "Italy",
    "Greater Pavia Metropolitan Area": "Italy",
    "Greater Parma Metropolitan Area": "Italy",
    "Greater Naples Metropolitan Area": "Italy",
    "Greater Bergamo Metropolitan Area": "Italy",
    "Greater Brescia Metropolitan Area": "Italy",
    "Greater Verona Metropolitan Area": "Italy",
    "Greater Rennes Metropolitan Area": "France",
    "Greater Grenoble Metropolitan Area": "France",
    "Greater Bordeaux Metropolitan Area": "France",
    "Greater Montpellier Metropolitan Area": "France",
    "Greater Strasbourg Metropolitan Area": "France",
    "Greater Toulouse Metropolitan Area": "France",
    "Greater Nantes Metropolitan Area": "France",
    "Greater Malmö Metropolitan Area": "Sweden",
    "Greater Gothenburg Metropolitan Area": "Sweden",
    "Greater Helsingborg Metropolitan Area": "Sweden",
    "Greater Västerås Metropolitan Area": "Sweden",
    "Tarnow Metropolitan Area": "Poland",
    "Cracow Metropolitan Area": "Poland",
    "Kielce Metropolitan Area": "Poland",
    "Katowice Metropolitan Area": "Poland",
    "Lodz Metropolitan Area": "Poland",
    "Poznan Metropolitan Area": "Poland",
    "Zamosc Metropolitan Area": "Poland",
    "Greater Palma de Mallorca Metropolitan Area": "Spain",
    "Greater Bilbao Metropolitan Area": "Spain",
    "Greater Vigo Metropolitan Area": "Spain",
    "Greater Granada Metropolitan Area": "Spain",
    "Greater Valencia Metropolitan Area": "Spain",
    "Galway Metropolitan Area": "Ireland",
    "Kortrijk Metropolitan Area": "Belgium",
    "Antwerp Metropolitan Area": "Belgium",
    "Ghent Metropolitan Area": "Belgium",
    "Charleroi Metropolitan Area": "Belgium",
    "Greater Nuremberg Metropolitan Area": "Germany",
    "Pecs Metropolitan Area": "Hungary",
    "Ljubljana Metropolitan Area": "Slovenia",
    "Hannover-Braunschweig-Göttingen-Wolfsburg Region": "Germany",
    "Brabantine City Row": "Netherlands",
    "Greater Dusseldorf Area": "Germany",
    "Greater Mulhouse Area": "France",
    "Linz-Wels-Steyr Area": "Austria",
    "Greater Alicante Area": "Spain",
    "Greater Funchal Area": "Portugal",
    "Greater Ipswich Area": "United Kingdom",
    "Greater La Coruña Area": "Spain",
    "Eindhoven Area": "Netherlands",
    "Greater Liverpool Area": "United Kingdom",
    "Greater Kecskemet Area": "Hungary",
    "Greater Leipzig Area": "Germany",
    "Utrecht Area": "Netherlands",
    "Greater Kiel Area": "Germany",
    "Greater Lyon Area": "France",
    "Greater Bielefeld Area": "Germany",
    "Greater Avignon Area": "France",
    "Greater Bern Area": "Switzerland",
    "Greater Hamburg Area": "Germany",
    "Greater Leeds Area": "United Kingdom",
    "Greater Metz Area": "France",
    "Greater Trencin Area": "Slovakia",
    "Osnabrück Land": "Germany",
    "Greater Salzburg": "Austria",
    "Greater Sankt Polten": "Austria",
    "Greater Graz": "Austria",
    "Greater Dublin": "Ireland",
    "Greater Nottingham": "United Kingdom",
    "Greater Munich Metropolitan Area": "Germany",
    "Belfast Metropolitan Area": "United Kingdom",
    "Belgrade Metropolitan Area": "Serbia",
    "Berlin Metropolitan Area": "Germany",
    "Bialystok Metropolitan Area": "Poland",
    "Bielsko-Biała Metropolitan Area": "Poland",
    "Blackpool Area": "United Kingdom",
    "Brasov Metropolitan Area": "Romania",
    "Bratislava Metropolitan Area": "Slovakia",
    "Bruges Metropolitan Area": "Belgium",
    "Budapest Metropolitan Area": "Hungary",
    "Burgas Metropolitan Area": "Bulgaria",
    "Bydgoszcz Metropolitan Area": "Poland",
    "Cluj-Napoca Metropolitan Area": "Romania",
    "Coimbra Metropolitan Area": "Portugal",
    "Czestochowa Metropolitan Area": "Poland",
    "Drecht Cities": "Netherlands",
    "Frankfurt Rhine-Main Metropolitan Area": "Germany",
    "Greater Aalborg Area": "Denmark",
    "Greater Aarhus Area": "Denmark",
    "Greater Almería Metropolitan Area": "Spain",
    "Greater Angers Area": "France",
    "Greater Annecy Area": "France",
    "Greater Asti Metropolitan Area": "Italy",
    "Greater Banska Bystrica Area": "Slovakia",
    "Greater Bari Metropolitan Area": "Italy",
    "Greater Blackburn with Darwen Area": "United Kingdom",
    "Greater Bologna Metropolitan Area": "Italy",
    "Greater Braga Area": "Portugal",
    "Greater Bremen Area": "Germany",
    "Greater Brest Area": "France",
    "Greater Brighton and Hove Area": "United Kingdom",
    "Greater Burgos Metropolitan Area": "Spain",
    "Greater Caen Area": "France",
    "Greater Cambridge Area": "United Kingdom",
    "Greater Cardiff Area": "United Kingdom",
    "Greater Cartagena Metropolitan Area": "Spain",
    "Greater Caserta Metropolitan Area": "Italy",
    "Greater Castellón de la Plana Area": "Spain",
    "Greater Ceske Budejovice Area": "Czechia",
    "Greater Chemnitz Area": "Germany",
    "Greater Cheshire West and Chester Area": "United Kingdom",
    "Greater Chomutov Area": "Czechia",
    "Greater Clermont-Ferrand Area": "France",
    "Greater Colmar Area": "France",
    "Greater Como Metropolitan Area": "Italy",
    "Greater Coventry Area": "United Kingdom",
    "Greater Cádiz Metropolitan Area": "Spain",
    "Greater Dijon Area": "France",
    "Greater Dresden Area": "Germany",
    "Greater Dunkerque Area": "France",
    "Greater Edinburgh Area": "United Kingdom",
    "Greater Enschede Area": "Netherlands",
    "Greater Erfurt Area": "Germany",
    "Greater Exeter Area": "United Kingdom",
    "Greater Faro Area": "Portugal",
    "Greater Flensburg Area": "Germany",
    "Greater Freiburg Area": "Germany",
    "Greater Genoa Metropolitan Area": "Italy",
    "Greater Gijón Metropolitan Area": "Spain",
    "Greater Girona Area": "Spain",
    "Greater Glasgow Area": "United Kingdom",
    "Greater Heilbronn Region": "Germany",
    "Greater Huelva Metropolitan Area": "Spain",
    "Greater Innsbruck": "Austria",
    "Greater Jaén Metropolitan Area": "Spain",
    "Greater Jihlava Area": "Czechia",
    "Greater Jyvaskyla Area": "Finland",
    "Greater Jönköping Metropolitan Area": "Sweden",
    "Greater Kaiserslautern Area": "Germany",
    "Greater Karlsruhe Area": "Germany",
    "Greater Kassel Area": "Germany",
    "Greater Kempten Area": "Germany",
    "Greater Koblenz Area": "Germany",
    "Greater Kuopio Area": "Finland",
    "Greater La Spezia Metropolitan Area": "Italy",
    "Greater Lahti Area": "Finland",
    "Greater Larisa Area": "Greece",
    "Greater Las Palmas Metropolitan Area": "Spain",
    "Greater Le Mans Area": "France",
    "Greater Lecce Metropolitan Area": "Italy",
    "Greater Lecco Metropolitan Area": "Italy",
    "Greater Leicester Area": "United Kingdom",
    "Greater Lerida Area": "Spain",
    "Greater Liberec Area": "Czechia",
    "Greater Lille Metropolitan Area": "France",
    "Greater Massa Metropolitan Area": "Italy",
    "Greater Mataró Metropolitan Area": "Spain",
    "Greater Modena Metropolitan Area": "Italy",
    "Greater Montbeliard Area": "France",
    "Greater Munster Area": "Germany",
    "Greater Murcia Metropolitan Area": "Spain",
    "Greater Málaga Metropolitan Area": "Spain",
    "Greater Nancy Area": "France",
    "Greater Nice Metropolitan Area": "France",
    "Greater Norrköping Metropolitan Area": "Sweden",
    "Greater Novara Metropolitan Area": "Italy",
    "Greater Nyiregyhaza Area": "Hungary",
    "Greater Odense Area": "Denmark",
    "Greater Olomouc Area": "Czechia",
    "Greater Orense Area": "Spain",
    "Greater Osijek Area": "Croatia",
    "Greater Oulu Area": "Finland",
    "Greater Padova (Padua) Metropolitan Area": "Italy",
    "Greater Palencia Metropolitan Area": "Spain",
    "Greater Palermo Metropolitan Area": "Italy",
    "Greater Pamplona Area": "Spain",
    "Greater Pardubice Area": "Czechia",
    "Greater Patras Area": "Greece",
    "Greater Pau Area": "France",
    "Greater Piacenza Metropolitan Area": "Italy",
    "Greater Pisa Metropolitan Area": "Italy",
    "Greater Plymouth Area": "United Kingdom",
    "Greater Ponta Delgada Area": "Portugal",
    "Greater Portsmouth Area": "United Kingdom",
    "Greater Prato Metropolitan Area": "Italy",
    "Greater Póvoa de Varzim Area": "Portugal",
    "Greater Reading Area": "United Kingdom",
    "Greater Reggio Emilia Metropolitan Area": "Italy",
    "Greater Rijeka Area": "Croatia",
    "Greater Rome Metropolitan Area": "Italy",
    "Greater Rostock Region": "Germany",
    "Greater Rouen Metropolitan Area": "France",
    "Greater Sabadell Metropolitan Area": "Spain",
    "Greater Saint-Brieuc Area": "France",
    "Greater Saint-Etienne Metropolitan Area": "France",
    "Greater Saint-Nazaire Area": "France",
    "Greater San Sebastian Area": "Spain",
    "Greater Santa Cruz de Tenerife Metropolitan Area": "Spain",
    "Greater Santander Metropolitan Area": "Spain",
    "Greater Santiago de Compostela Metropolitan Area": "Spain",
    "Greater Sassari Metropolitan Area": "Italy",
    "Greater Sheffield Area": "United Kingdom",
    "Greater Slavonski Brod Area": "Croatia",
    "Greater Split Area": "Croatia",
    "Greater Stara Zagora Area": "Bulgaria",
    "Greater Stoke-on-Trent Area": "United Kingdom",
    "Greater Suceava Botosani Area": "Romania",
    "Greater Swansea Area": "United Kingdom",
    "Greater Szeged Area": "Hungary",
    "Greater Szombathely Area": "Hungary",
    "Greater Taranto Metropolitan Area": "Italy",
    "Greater Tarragona Area": "Spain",
    "Greater Tartu Area": "Estonia",
    "Greater Toulon Metropolitan Area": "France",
    "Greater Tours Area": "France",
    "Greater Trento Metropolitan Area": "Italy",
    "Greater Treviso Metropolitan Area": "Italy",
    "Greater Trieste Metropolitan Area": "Italy",
    "Greater Trnava Area": "Slovakia",
    "Greater Tubingen Area": "Germany",
    "Greater Udine Metropolitan Area": "Italy",
    "Greater Umeå Metropolitan Area": "Sweden",
    "Greater Uppsala Metropolitan Area": "Sweden",
    "Greater Usti nad Labem Area": "Czechia",
    "Greater Valladolid Metropolitan Area": "Spain",
    "Greater Varese Metropolitan Area": "Italy",
    "Greater Venice Metropolitan Area": "Italy",
    "Greater Vicenza Metropolitan Area": "Italy",
    "Greater Vitoria Area": "Spain",
    "Greater Volos Area": "Greece",
    "Greater Wurzburg Area": "Germany",
    "Greater Zadar Area": "Croatia",
    "Greater Zilina Area": "Slovakia",
    "Greater Örebro Metropolitan Area": "Sweden",
    "Helsinki Metropolitan Area": "Finland",
    "Iasi Metropolitan Area": "Romania",
    "Klagenfurt-Villach Area": "Austria",
    "Kosice Metropolitan Area": "Slovakia",
    "Liege Metropolitan Area": "Belgium",
    "Limerick Metropolitan Area": "Ireland",
    "Louvain Metropolitan Area": "Belgium",
    "Lublin Metropolitan Area": "Poland",
    "Maribor Metropolitan Area": "Slovenia",
    "Miskolc Metropolitan Area": "Hungary",
    "Mons Metropolitan Area": "Belgium",
    "Namur Metropolitan Area": "Belgium",
    "Novi Sad Metropolitan Area": "Serbia",
    "Ostend Metropolitan Area": "Belgium",
    "Ostrava Metropolitan Area": "Czechia",
    "Ploiesti Metropolitan Area": "Romania",
    "Ruhr Region": "Germany",
    "Rzeszow Metropolitan Area": "Poland",
    "Satu Mare Metropolitan Area": "Romania",
    "Spain Area": "Spain",
    "Spain Metropolitan Area": "Spain",
    "Szczecin Metropolitan Area": "Poland",
    "Szekesfehervar Metropolitan Area": "Hungary",
    "Tallinn Metropolitan Area": "Estonia",
    "Timisoara Metropolitan Area": "Romania",
    "Trier Region": "Germany",
    "Turku Metropolitan Area": "Finland",
    "Tyneside Area": "United Kingdom",
    "Varna Metropolitan Area": "Bulgaria",
    "Waterford Metropolitan Area": "Ireland",
    "Wroclaw Metropolitan Area": "Poland",
    "Zagreb Metropolitan Area": "Croatia",
    "Zielona Gora Metropolitan Area": "Poland",
    "s-Hertogenbosch Area": "Netherlands",
    "Åland Islands": "Finland",
}

NON_COUNTRY_LOCATION_PATTERN = re.compile(
    r'EMEA|European Union|European Economic Area',
    flags=re.IGNORECASE,
)

SKILL_PREFIX_PATTERN = re.compile(r'^\s*skills:\s*', flags=re.IGNORECASE)
SKILL_SUFFIX_MORE_PATTERN = re.compile(r'\s*,\s*\+\s*\d+\s+more\s*$', flags=re.IGNORECASE)
SKILL_PROFILE_MATCH_PATTERN = re.compile(
    r'^\s*\d+\s+of\s+\d+\s+skills\s+match\s+your\s+profile\s*-\s*you\s+may\s+be\s+a\s+good\s+fit\s*$',
    flags=re.IGNORECASE,
)
CTA_SENTENCE_PATTERN = re.compile(
    r'(?i)(^|[.!?\n])\s*[^.!?\n]*\b(?:'
    r'why\s+join\s+us\??|'
    r'why\s+you\s+(?:should|sould)\s+join\s+us\??|'
    r'(?:apply|appy)\s+and\s+join\s+us|'
    r'so\s+join\s+us|'
    r'join\s+us|'
    r'want\s+to\s+join\s+us\??|'
    r'contact\s+us(?:\s+at)?|'
    r'apply\s+now|apply\s+today|apply\s+here'
    r')(?=\s|$|[.!?\-]|[A-Z])[^.!?\n]*[.!?\n]'
)

SECTION_HEADER_SPLIT_PATTERN = re.compile(
    r'(?i)\b(?:'
    r'about\s+us|who\s+are\s+we|company\s+description|job\s+description|'
    r'responsibilities?|requirements?|qualifications?|benefits?|what\s+we\s+offer|'
    r'our\s+mission|our\s+values|how\s+to\s+apply|application\s+process|'
    r'to\s+apply|'
    r'recruitment\s+process|selection\s+process|our\s+selection\s+process|'
    r'additional\s+information|profile\s+sought|desired\s+profile'
    r')\b\s*[:.]'
)

BULLET_MARKER_SPLIT_PATTERN = re.compile(r'\s*[\u2022\u25cf\u25aa\u25e6\u25c9]\s*')
DASH_BULLET_PREFIX_PATTERN = re.compile(r'^\s*[-*]\s+')

DESCRIPTION_BLOCK_FILTER_PATTERNS = {
    'Company intro heading': re.compile(
        r'(?i)^\s*(?:who\s+are\s+we\??|about\s+us|company\s+description)\s*[:.!?]*\s*$'
    ),
    'Equal opportunity / anti-discrimination': re.compile(
        r'(?i)\b(?:equal\s+opportunit(?:y|ies)|non[- ]?discrimination|'
        r'diversity\s+and\s+inclusion|all\s+qualified\s+applicants|'
        r'regardless\s+of\s+(?:age|gender|race|ethnicity|religion|disability|sexual\s+orientation|nationality)|'
        r'both\s+sexes|all\s+ages\s+and\s+all\s+nationalities|protected\s+veteran)\b'
    ),
    'GDPR/privacy/legal disclaimer': re.compile(
        r'(?i)\b(?:gdpr|data\s+protection|privacy\s+policy|processing\s+of\s+personal\s+data|'
        r'art\.?\s*13\b|regulation\s*\(?\s*(?:eu\s*)?2016\s*/\s*679\s*\)?|cookie(?:s)?)\b'
    ),
    'Application process boilerplate': re.compile(
        r'(?i)\b(?:apply\s+now|how\s+to\s+apply|application\s+process|recruitment\s+process|'
        r'selection\s+process|our\s+selection\s+process|submit\s+your\s+application|'
        r'click\s+(?:on\s+)?(?:the\s+)?apply|interview\s+process|interview\s+rounds?|'
        r'to\s+apply|please\s+submit\s+your\s+cv|please\s+send\s+(?:through\s+)?(?:an\s+updated\s+)?cv|'
        r'send\s+your\s+cv|send\s+us\s+your\s+application|if\s+you\s+are\s+(?:suitable\s+and\s+)?interested|'
        r'please\s+visit\s+our\s+website|visit\s+our\s+website|for\s+more\s+information\s+please\s+contact|'
        r'please\s+contact\s+your\s+recruiting\s+partner|please\s+get\s+in\s+touch\s+with\s+the\s+person\s+responsible|'
        r'if\s+this\s+is\s+an\s+interest\s+please\s+reach\s+out\s+to\s+me\s+on\s+the\s+below\s+details|'
        r'please\s+contact\s+us\s+as\s+soon\s+as\s+possible|please\s+see\s+below\s+for\s+what\s+we\'re\s+looking\s+for|'
        r'what\s+are\s+you\s+waiting\s+for\??\s*join\s+us|thank\s+you\s*!?)\b'
    ),
    'Recruiter contact block': re.compile(
        r'(?i)\b(?:contact\s+person|your\s+contact|for\s+more\s+information(?:,)?\s+please\s+contact|'
        r'please\s+contact|reach\s+out\s+to|recruit(?:ing|er)?\s+partner)\b.*'
        r'(?:@|linkedin|tel(?:ephone)?|phone|\+?\d[\d\s().\-]{6,}|cv|resume)'
    ),
    'Company marketing paragraph': re.compile(
        r'(?i)\b(?:about\s+us|who\s+are\s+we|our\s+mission|our\s+values|'
        r'founded\s+in\s+\d{4}|we\s+are\s+(?:a|an|the)\s+(?:leading|global|world\s+leader)|'
        r'with\s+over\s+\d[\d,.\s]*(?:employees|people|professionals)|great\s+place\s+to\s+work|'
        r'join\s+our\s+team|join\s+us|make\s+an\s+impact)\b'
    ),
}


def _init_block_filter_stats():
    """Create an empty stats object for description block filtering."""
    return {
        'records_with_removed_blocks': 0,
        'blocks_removed': 0,
        'removed_by_category': {
            category: 0
            for category in DESCRIPTION_BLOCK_FILTER_PATTERNS
        },
    }


def split_description_into_blocks(text):
    """Split description text into blocks using line breaks, bullets, and section headers."""
    if not text:
        return []

    normalized = text.replace('\r\n', '\n').replace('\r', '\n')

    # Create new block boundaries before common section headers when they appear inline.
    normalized = SECTION_HEADER_SPLIT_PATTERN.sub(lambda match: f"\n{match.group(0)}", normalized)

    blocks = []
    for raw_line in normalized.split('\n'):
        line = raw_line.strip()
        if not line:
            continue

        line = DASH_BULLET_PREFIX_PATTERN.sub('', line).strip()
        if not line:
            continue

        bullet_parts = [part.strip() for part in BULLET_MARKER_SPLIT_PATTERN.split(line) if part.strip()]
        if bullet_parts:
            blocks.extend(bullet_parts)

    return blocks


def _get_block_filter_category(block):
    """Return the first matching block-filter category, or None if block should be kept."""
    for category, pattern in DESCRIPTION_BLOCK_FILTER_PATTERNS.items():
        if category == 'Company marketing paragraph' and len(block) < 120:
            continue

        if pattern.search(block):
            return category

    return None


def clean_description_blocks(df, column='Description'):
    """Drop block-level boilerplate noise from descriptions and report category stats."""
    cleaned_df = df.copy()
    stats = _init_block_filter_stats()

    if column not in cleaned_df.columns:
        return cleaned_df, stats

    cleaned_values = []
    for value in cleaned_df[column]:
        if pd.isna(value):
            cleaned_values.append(value)
            continue

        text = str(value)
        blocks = split_description_into_blocks(text)
        if not blocks:
            cleaned_values.append(text)
            continue

        kept_blocks = []
        removed_in_record = 0

        for block in blocks:
            category = _get_block_filter_category(block)
            if category is None:
                kept_blocks.append(block)
                continue

            removed_in_record += 1
            stats['blocks_removed'] += 1
            stats['removed_by_category'][category] += 1

        if removed_in_record > 0:
            stats['records_with_removed_blocks'] += 1

        cleaned_values.append('\n'.join(kept_blocks).strip())

    cleaned_df[column] = cleaned_values
    return cleaned_df, stats

def get_language_check(df, field='Description', mode='sample', sample_size=500):
    """Reusable language check for preprocessing decisions."""
    if field not in df.columns:
        return None
    return detect_language_distribution(
        df[field],
        mode=mode,
        sample_size=sample_size,
    )


def remove_records_with_all_critical_fields_invalid(df):
    """
    Detect and remove job-posting records where every critical field is invalid.

    A record is considered invalid only when all existing critical fields are invalid
    at the same time. This keeps rows that still contain useful information in at
    least one important field.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, list[str]]:
            - cleaned_df: DataFrame after removing fully invalid records
            - invalid_records: DataFrame containing only the removed records
            - checked_fields: critical fields that were actually present and checked
    """
    invalid_mask, checked_fields = get_all_critical_fields_invalid_mask(df)

    # If none of the expected critical fields exist, there is nothing to filter.
    if invalid_mask is None:
        return df.copy(), df.iloc[0:0].copy(), checked_fields

    # Keep a separate copy of removed rows for reporting and debugging.
    invalid_records = df[invalid_mask].copy()

    # Keep only rows that still have at least one usable critical field.
    cleaned_df = df[~invalid_mask].copy()

    return cleaned_df, invalid_records, checked_fields


def drop_unneeded_job_posting_columns(df):
    """Remove columns that are not needed for this analysis.
    The function only drops columns that actually exist in the input DataFrame. """
    existing_columns_to_drop = [
        column for column in JOB_POSTINGS_COLUMNS_TO_DROP if column in df.columns
    ]

    if not existing_columns_to_drop:
        return df.copy(), []

    cleaned_df = df.drop(columns=existing_columns_to_drop).copy()
    return cleaned_df, existing_columns_to_drop


def print_sample_record(df, label, max_fields=10, max_text_length=120):
    """Print a compact sample record to show how the data looks at a given step."""
    print(f"\n  {label}")

    if df.empty:
        print("    No records available")
        return

    record = df.iloc[0].to_dict()
    preview = {}
    for index, (key, value) in enumerate(record.items()):
        if index >= max_fields:
            break

        if isinstance(value, str) and len(value) > max_text_length:
            preview[key] = f"{value[:max_text_length]}..."
        else:
            preview[key] = value

    print(f"    {preview}")


def clean_markup_from_text(value):
    """Remove markup/contact details and normalize text for downstream NLP."""
    if pd.isna(value):
        return value

    text = html.unescape(str(value))
    text = MARKDOWN_LINK_PATTERN.sub(r'\1', text)
    text = RAW_URL_PATTERN.sub(' ', text)
    text = HTML_TAG_PATTERN.sub(' ', text)
    text = MARKDOWN_BOLD_PATTERN.sub(r'\1', text)
    text = MARKDOWN_UNDERSCORE_PATTERN.sub(r'\1', text)
    text = INLINE_CODE_PATTERN.sub(r'\1', text)
    text = remove_email_addresses(text)
    text = remove_phone_numbers(text)
    text = remove_broken_or_partial_urls(text)
    text = remove_asterisk_clusters(text)
    text = remove_redacted_placeholders(text)
    text = remove_empty_wrappers(text)
    text = remove_gender_marker_tokens(text)
    text = QUOTE_CHARACTER_PATTERN.sub('', text)
    text = DECORATIVE_SEPARATOR_PATTERN.sub('', text)
    text = remove_emoji_like_unicode(text)
    text = CTA_SENTENCE_PATTERN.sub(r'\1', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    text = NEWLINE_SPACING_PATTERN.sub('\n', text)
    text = COLLAPSE_NEWLINES_PATTERN.sub('\n', text)
    text = NEWLINE_TO_PERIOD_PATTERN.sub('. ', text)
    return text.strip()


def clean_gender_markers_in_columns(df, columns=('Title', 'Description')):
    """Remove gender marker variants from selected text columns and return per-column counts."""
    cleaned_df = df.copy()
    cleaned_counts = {}

    for column in columns:
        if column not in cleaned_df.columns:
            continue

        texts = cleaned_df[column].fillna('').astype(str)
        marker_mask = texts.str.contains(GENDER_MARKER_PATTERN, regex=True)
        cleaned_counts[column] = int(marker_mask.sum())

        if cleaned_counts[column] == 0:
            continue

        cleaned_df.loc[marker_mask, column] = texts[marker_mask].apply(
            lambda text: WHITESPACE_PATTERN.sub(' ', remove_gender_marker_tokens(text)).strip()
        )

    return cleaned_df, cleaned_counts


def add_country_from_location(df, source_column='Location', target_column='Country'):
    """Add a country column extracted from the location text."""
    cleaned_df = df.copy()

    if source_column not in cleaned_df.columns:
        cleaned_df[target_column] = pd.NA
        return cleaned_df

    cleaned_df[source_column] = cleaned_df[source_column].replace(LOCATION_TO_COUNTRY_OVERRIDES)
    def _extract_country(value):
        if not isinstance(value, str):
            return pd.NA

        if NON_COUNTRY_LOCATION_PATTERN.search(value):
            return pd.NA

        if ',' in value:
            return value.split(',')[-1].strip()

        return value.strip()

    cleaned_df[target_column] = cleaned_df[source_column].apply(_extract_country)
    return cleaned_df


def add_work_modality_from_primary_description(
    df,
    source_column='Primary Description',
    target_column='Work Modality',
):
    """Extract work modality (remote/hybrid/on-site) from the last parenthetical in Primary Description."""
    cleaned_df = df.copy()

    if source_column not in cleaned_df.columns:
        cleaned_df[target_column] = pd.NA
        return cleaned_df, 0, len(cleaned_df)

    pattern = re.compile(r'\(([^)]*)\)\s*$')
    filled_count = 0
    missing_count = 0
    values = []

    for value in cleaned_df[source_column]:
        if not isinstance(value, str):
            values.append(pd.NA)
            missing_count += 1
            continue

        match = pattern.search(value.strip())
        if not match:
            values.append(pd.NA)
            missing_count += 1
            continue

        content = match.group(1).strip().lower()

        if 'remote' in content:
            values.append('remote')
            filled_count += 1
        elif 'hybrid' in content:
            values.append('hybrid')
            filled_count += 1
        elif 'on-site' in content or 'on site' in content or 'onsite' in content:
            values.append('on-site')
            filled_count += 1
        else:
            values.append(pd.NA)
            missing_count += 1

    cleaned_df[target_column] = values
    return cleaned_df, filled_count, missing_count


def add_company_from_primary_description(
    df,
    source_column='Primary Description',
    target_column='Company',
    delimiter='·',
):
    """Extract company name from Primary Description up to the first delimiter."""
    cleaned_df = df.copy()

    if source_column not in cleaned_df.columns:
        cleaned_df[target_column] = pd.NA
        return cleaned_df

    values = []
    for value in cleaned_df[source_column]:
        if not isinstance(value, str):
            values.append(pd.NA)
            continue

        parts = value.split(delimiter, 1)
        if len(parts) < 2:
            values.append(pd.NA)
            continue

        company = parts[0].strip()
        values.append(company if company else pd.NA)

    cleaned_df[target_column] = values
    return cleaned_df


def _clean_skill_value(value):
    """Normalize Skill text by removing boilerplate wrappers and profile-match sentences."""
    if pd.isna(value):
        return value

    text = str(value).strip()

    if SKILL_PROFILE_MATCH_PATTERN.fullmatch(text):
        return ''

    text = SKILL_PREFIX_PATTERN.sub('', text)
    text = SKILL_SUFFIX_MORE_PATTERN.sub('', text)
    return WHITESPACE_PATTERN.sub(' ', text).strip()


def clean_skill_feature(df, column='Skill'):
    """Clean the Skill column and return per-rule counts."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        return cleaned_df, {
            'prefix_removed': 0,
            'suffix_removed': 0,
            'profile_match_cleared': 0,
        }

    texts = cleaned_df[column].fillna('').astype(str)
    stats = {
        'prefix_removed': int(texts.str.contains(SKILL_PREFIX_PATTERN, regex=True).sum()),
        'suffix_removed': int(texts.str.contains(SKILL_SUFFIX_MORE_PATTERN, regex=True).sum()),
        'profile_match_cleared': int(texts.str.contains(SKILL_PROFILE_MATCH_PATTERN, regex=True).sum()),
    }

    cleaned_df[column] = cleaned_df[column].apply(_clean_skill_value)
    return cleaned_df, stats


def normalize_invalid_to_missing(df, columns):
    """Replace invalid text values with missing markers and return counts per column."""
    cleaned_df = df.copy()
    counts = {}

    for column in columns:
        if column not in cleaned_df.columns:
            continue

        invalid_mask = invalid_content_mask(cleaned_df[column])
        counts[column] = int(invalid_mask.sum())

        if counts[column] == 0:
            continue

        cleaned_df.loc[invalid_mask, column] = pd.NA

    return cleaned_df, counts


def save_markup_cleaning_examples(before_records, after_df, detection_result, before_path, after_path, column='Description', sample_count=20):
    """Save before/after examples for records whose descriptions contained markup-like content."""
    sample_before = before_records.head(sample_count)
    before_export = []
    after_export = []

    for index, record in sample_before.iterrows():
        detected_types = get_detected_markup_types(index, detection_result)
        before_export.append({
            'Title': record.get('Title'),
            'Location': record.get('Location'),
            'Primary Description': record.get('Primary Description'),
            'Description': record.get(column),
            'Detected Markup Types': detected_types,
        })

        after_record = after_df.loc[index]
        after_export.append({
            'Title': after_record.get('Title'),
            'Location': after_record.get('Location'),
            'Primary Description': after_record.get('Primary Description'),
            'Description': after_record.get(column),
            'Detected Markup Types Before Cleaning': detected_types,
        })

    if before_path is not None:
        with open(before_path, 'w', encoding='utf-8') as before_file:
            json.dump(before_export, before_file, ensure_ascii=False, indent=2)

    if after_path is not None:
        with open(after_path, 'w', encoding='utf-8') as after_file:
            json.dump(after_export, after_file, ensure_ascii=False, indent=2)


def clean_description_markup(df, column='Description', before_examples_path=None, after_examples_path=None):
    """
    Detect and remove markup-like content from the description column.

    This step cleans URLs, HTML tags/entities, and simple markdown-like patterns
    while preserving the remaining readable text.
    """
    if column not in df.columns:
        return df.copy(), df.iloc[0:0].copy(), 0, _init_block_filter_stats()

    markup_records, detection_result = find_records_with_markup(df, column=column)
    cleaned_df, block_filter_stats = clean_description_blocks(df, column=column)
    cleaned_df[column] = cleaned_df[column].apply(clean_markup_from_text)

    remaining_markup_records, _ = find_records_with_markup(cleaned_df, column=column)

    if before_examples_path is not None or after_examples_path is not None:
        save_markup_cleaning_examples(
            markup_records,
            cleaned_df,
            detection_result,
            before_examples_path,
            after_examples_path,
            column=column,
        )

    return cleaned_df, markup_records, len(remaining_markup_records), block_filter_stats


def preprocess_ecsf(data):
    """
    Preprocess ECSF data
    """
    print("🔧 Preprocessing ECSF data...")
    def _preview_record(record, max_text_length=160):
        if not isinstance(record, dict):
            return record

        preview = {}
        for key, value in record.items():
            if isinstance(value, str) and len(value) > max_text_length:
                preview[key] = f"{value[:max_text_length]}..."
            else:
                preview[key] = value
        return preview

    def _print_structure_sample(label, payload):
        if not isinstance(payload, dict):
            print(f"  {label}: non-dict payload")
            return

        top_keys = sorted(payload.keys())
        print(f"  {label} top-level keys: {top_keys}")

        work_roles = payload.get('work_role')
        if isinstance(work_roles, list) and work_roles:
            print(f"  {label} work_role[0] keys: {sorted(work_roles[0].keys())}")
            print(f"  {label} work_role[0] sample: {_preview_record(work_roles[0])}")

        tks_items = payload.get('tks')
        if isinstance(tks_items, list) and tks_items:
            print(f"  {label} tks[0] keys: {sorted(tks_items[0].keys())}")
            print(f"  {label} tks[0] sample: {_preview_record(tks_items[0])}")

        relationships = payload.get('relationship')
        if isinstance(relationships, list) and relationships:
            print(f"  {label} relationship[0] keys: {sorted(relationships[0].keys())}")
            print(f"  {label} relationship[0] sample: {_preview_record(relationships[0])}")

    def _remove_task_tks(payload):
        if not isinstance(payload, dict):
            return payload

        tks_items = payload.get('tks')
        if not isinstance(tks_items, list):
            return payload

        kept_tks = [item for item in tks_items if item.get('type') != 'Task']
        removed_count = len(tks_items) - len(kept_tks)
        kept_ids = {item.get('id') for item in kept_tks if item.get('id') is not None}

        payload = payload.copy()
        payload['tks'] = kept_tks

        relationships = payload.get('relationship')
        removed_relationships = 0
        if isinstance(relationships, list):
            kept_relationships = [
                item for item in relationships
                if item.get('tks_id') in kept_ids
            ]
            removed_relationships = len(relationships) - len(kept_relationships)
            payload['relationship'] = kept_relationships

        print(
            "  ECSF Task filtering: "
            f"tasks_removed={removed_count}, relationships_removed={removed_relationships}"
        )
        return payload

    def _remove_work_role_fields(payload, fields_to_remove):
        if not isinstance(payload, dict):
            return payload, 0

        work_roles = payload.get('work_role')
        if not isinstance(work_roles, list):
            return payload, 0

        removed_count = 0
        cleaned_roles = []
        for role in work_roles:
            if not isinstance(role, dict):
                cleaned_roles.append(role)
                continue

            role_copy = role.copy()
            for field in fields_to_remove:
                if field in role_copy:
                    role_copy.pop(field, None)
                    removed_count += 1

            cleaned_roles.append(role_copy)

        payload = payload.copy()
        payload['work_role'] = cleaned_roles
        return payload, removed_count

    def _normalize_apostrophes(value):
        if isinstance(value, str):
            return value.replace('\\u2019', '\u2019')
        if isinstance(value, list):
            return [_normalize_apostrophes(item) for item in value]
        if isinstance(value, dict):
            return {key: _normalize_apostrophes(item) for key, item in value.items()}
        return value

    _print_structure_sample('Initial', data)
    normalized = _normalize_apostrophes(data)
    normalized, removed_fields = _remove_work_role_fields(
        normalized,
        fields_to_remove={'summary_statement', 'mission'},
    )
    print(f"  ECSF work_role fields removed: {removed_fields}")
    normalized = _remove_task_tks(normalized)
    _print_structure_sample('Final', normalized)
    return normalized


def preprocess_job_postings(data):
    """
    Preprocess Job Postings data
    - Clean text fields
    - Extract skills/requirements text
    - Handle missing values
    - Standardize location data
    - Remove records with all critical fields invalid
    """
    print("🔧 Preprocessing Job Postings data...")

    df = pd.DataFrame(data)
    print(f"  Loaded {len(df)} job posting records")
    print_sample_record(df, 'Sample record at start:')

    # Step 1: drop columns that are not needed for the current pipeline stage.
    # Doing this first simplifies the record shape for all following preprocessing steps.
    cleaned_df, dropped_columns = drop_unneeded_job_posting_columns(df)

    if dropped_columns:
        print(f"  Dropped columns: {dropped_columns}")
        print(f"  Remaining columns: {len(cleaned_df.columns)}")
    else:
        print("  No configured columns found to drop")

    # Step 2: normalize invalid content values to missing markers.
    missing_columns = ['Description', 'Primary Description', 'Location', 'Skill']
    cleaned_df, missing_counts = normalize_invalid_to_missing(cleaned_df, missing_columns)
    if missing_counts:
        formatted_counts = ', '.join(
            f"{column}={count}" for column, count in missing_counts.items()
        )
        print(f"  Missing value replacements (invalid content): {formatted_counts}")
    else:
        print("  No target columns found for missing value normalization")

    # Step 3: detect markup records for reporting only (no before/after exports).
    markup_records, detection_result = find_records_with_markup(cleaned_df, column='Description')

    # Step 4: remove gender marker variants from title/description text.
    cleaned_df, gender_marker_counts = clean_gender_markers_in_columns(cleaned_df)
    if gender_marker_counts:
        for column, count in gender_marker_counts.items():
            print(f"  {column} records with gender markers removed: {count}")
    else:
        print("  No target columns found for gender marker cleaning")

    # Step 5: remove markup-like content from descriptions and save "after" examples.
    cleaned_df, markup_records_cleaned, remaining_markup_count, description_block_stats = clean_description_markup(
        cleaned_df,
        before_examples_path=None,
        after_examples_path=None,
    )
    print(
        "  Description block filtering: "
        f"records_with_removed_blocks={description_block_stats['records_with_removed_blocks']}, "
        f"blocks_removed={description_block_stats['blocks_removed']}"
    )
    removed_categories = [
        f"{category}={count}"
        for category, count in description_block_stats['removed_by_category'].items()
        if count > 0
    ]
    if removed_categories:
        print(f"    Removed by category: {', '.join(removed_categories)}")

    print(f"  Description records with markup details: detected={len(markup_records)}, remaining after cleaning={remaining_markup_count}")
    # Step 6: clean Skill feature boilerplate.
    cleaned_df, skill_clean_stats = clean_skill_feature(cleaned_df, column='Skill')
    if 'Skill' in cleaned_df.columns:
        print(
            "  Skill cleanup: "
            f"prefix_removed={skill_clean_stats['prefix_removed']}, "
            f"suffix_removed={skill_clean_stats['suffix_removed']}, "
            f"profile_match_cleared={skill_clean_stats['profile_match_cleared']}"
        )
    else:
        print("  Skill column not found for Skill cleanup")

    # Step 7: derive country from location (keep rows even if country is missing).
    cleaned_df = add_country_from_location(cleaned_df)
    if 'Location' in cleaned_df.columns and 'Country' in cleaned_df.columns:
        columns = list(cleaned_df.columns)
        location_index = columns.index('Location')
        columns.remove('Country')
        columns.insert(location_index + 1, 'Country')
        cleaned_df = cleaned_df[columns]

    # Step 8: derive work modality from primary description.
    cleaned_df, modality_filled, modality_missing = add_work_modality_from_primary_description(cleaned_df)
    if 'Primary Description' in cleaned_df.columns and 'Work Modality' in cleaned_df.columns:
        columns = list(cleaned_df.columns)
        primary_index = columns.index('Primary Description')
        columns.remove('Work Modality')
        columns.insert(primary_index + 1, 'Work Modality')
        cleaned_df = cleaned_df[columns]

    print(
        "  Work modality extraction: "
        f"filled={modality_filled}, missing={modality_missing}"
    )

    # Step 9: derive company from primary description.
    cleaned_df = add_company_from_primary_description(cleaned_df)
    if 'Description' in cleaned_df.columns and 'Company' in cleaned_df.columns:
        columns = list(cleaned_df.columns)
        description_index = columns.index('Description')
        columns.remove('Company')
        columns.insert(description_index + 1, 'Company')
        cleaned_df = cleaned_df[columns]

    # Step 10: remove rows where every critical field is invalid.
    cleaned_df, invalid_records, checked_fields = remove_records_with_all_critical_fields_invalid(cleaned_df)

    if checked_fields:
        print(f"  Critical fields checked: {checked_fields}")
        print(f"  Invalid records removed: {len(invalid_records)}")
        print(f"  Remaining records: {len(cleaned_df)}")
    else:
        print("  No critical fields found for invalid-record detection")

    print_sample_record(cleaned_df, 'Sample record at end:')

    return cleaned_df.to_dict('records')


def save_preprocessed_data(data, filename):
    """Save preprocessed data to JSON"""
    output_path = PREPROCESSED_DIR / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved: {output_path}")


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(description='Run job postings preprocessing in sample or full mode.')
    add_sample_mode_arguments(
        parser,
        mode_flag='--run-mode',
        mode_dest='run_mode',
        default_mode='full',
        mode_help="Execution mode: 'sample' for quick iteration or 'full' for complete dataset (default: full).",
        sample_size_default=1000,
        sample_size_help='Number of records used when --run-mode=sample (default: 1000).',
    )
    args = parser.parse_args()

    if not is_valid_sample_size(args.sample_size):
        print('❌ sample-size must be greater than 0')
        return

    print("⚙️  Starting Data Preprocessing...\n")
    print(f"  Run mode: {args.run_mode}")
    if args.run_mode == 'sample':
        print(f"  Sample size: {args.sample_size}")
    
    # Load raw data
    ecsf_file = RAW_DATA_DIR / 'ecsf.json'
    job_postings_file = RAW_DATA_DIR / 'job_postings.json'
    
    if not job_postings_file.exists():
        print("❌ Raw data files not found. Run explore_data.py first.")
        return
    
    # Load
    with open(ecsf_file, 'r', encoding='utf-8') as f:
        ecsf_data = json.load(f)
    
    with open(job_postings_file, 'r', encoding='utf-8') as f:
        job_data = json.load(f)

    original_job_count = len(job_data)
    job_data = sample_collection(job_data, mode=args.run_mode, sample_size=args.sample_size)
    print(f"  Job postings loaded: {len(job_data)} / {original_job_count}")
    
    ecsf_preprocessed = preprocess_ecsf(ecsf_data)
    job_preprocessed = preprocess_job_postings(job_data)

    # Save preprocessed data
    save_preprocessed_data(ecsf_preprocessed, 'ecsf_preprocessed.json')
    if args.run_mode == 'sample':
        output_name = f"job_postings_preprocessed_sample_{len(job_data)}.json"
    else:
        output_name = 'job_postings_preprocessed.json'

    save_preprocessed_data(job_preprocessed, output_name)


if __name__ == '__main__':
    main()
