"""
WhatsApp Chat Analyzer — Advanced Production-Grade Streamlit Application
=========================================================================
Comprehensive analytics, NLP, ML, and visualization for WhatsApp exports.

Run:
    streamlit run whatsapp_analyzer.py

Requirements:
    pip install streamlit pandas numpy plotly scikit-learn nltk vaderSentiment
                textblob langdetect wordcloud matplotlib scipy
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import re
import io
import logging
import traceback
import warnings
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Optional NLP / ML packages — degrade gracefully ──────────────────────────

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import nltk
    from nltk.util import ngrams as nltk_ngrams
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from langdetect import detect as lang_detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PAGE_TITLE = "SmartChat Intelligence Engine"
PAGE_ICON  = "💬"
PLOTLY_TPL = "plotly_dark"

# Extended timestamp patterns (global formats)
# \s matches regular space; [\s\u202f] also matches Narrow No-Break Space
# used by WhatsApp on some locales between time and am/pm
_SP = r"[\s\u202f]"   # space OR narrow no-break space

CHAT_PATTERNS = [
    # [DD/MM/YYYY, HH:MM:SS] Author: message  (iOS square brackets)
    r"\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s([^:]+):\s(.+)",
    # [DD/MM/YYYY, HH:MM:SS AM/PM] Author: message (iOS, with am/pm)
    r"\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}" + _SP + r"[AaPp][Mm])\]\s([^:]+):\s(.+)",
    # DD/MM/YY, HH:MM am/pm - Author: message  (Indian/Android — narrow-nbsp before am/pm)
    r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}" + _SP + r"[AaPp][Mm])\s-\s([^:]+):\s(.+)",
    # DD/MM/YYYY, HH:MM - Author: message  (EU Android, 24h)
    r"(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)",
    # YYYY-MM-DD HH:MM:SS - Author: message  (some exports)
    r"(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})\s-\s([^:]+):\s(.+)",
    # DD.MM.YYYY, HH:MM - Author: message  (German locale)
    r"(\d{1,2}\.\d{1,2}\.\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.+)",
]

DATETIME_FORMATS = [
    "%d/%m/%y %I:%M %p",    # DD/MM/YY HH:MM am/pm  ← Indian WhatsApp format
    "%d/%m/%Y %I:%M %p",    # DD/MM/YYYY HH:MM am/pm
    "%m/%d/%y %I:%M %p",
    "%m/%d/%Y %I:%M %p",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%d/%m/%y %H:%M",
    "%m/%d/%y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%y %I:%M:%S %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%y %H:%M",
]

ENCODING_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

STOP_WORDS = frozenset({
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "as", "i", "you", "me", "we",
    "media", "omitted", "image", "video", "this", "that", "it", "be",
    "have", "do", "will", "would", "could", "should", "am", "are",
    "was", "were", "has", "had", "he", "she", "they", "them", "his",
    "her", "our", "your", "my", "its", "so", "if", "then", "than",
    "from", "by", "up", "out", "no", "not", "just", "what", "when",
    "where", "who", "how", "all", "some", "one", "can", "about", "ok",
    "okay", "yeah", "yes", "lol", "haha", "ha", "na", "da", "la",
    "https", "http", "www", "com", "also", "get", "got", "like",
    "know", "think", "see", "come", "going", "want", "need", "make",
    "time", "day", "good", "well", "still", "even", "back", "way",
    "oh", "ah", "uh", "um", "ya", "yep", "nope", "hey", "hi", "bye",
    "re", "ve", "ll", "don", "won", "can", "isn", "aren", "wasn",
    "message", "deleted", "edited", "forwarded", "more", "much", "many",
    "really", "very", "too", "quite", "bit", "lot", "few", "any",
    "every", "never", "always", "often", "sometimes", "already", "yet",
    "now", "here", "there", "then", "only", "just", "also", "both",
    "each", "other", "same", "such", "own", "into", "through", "during",
    "before", "after", "above", "below", "again", "further", "once",
})

# System message patterns to discard
# FIX: Removed the `.+: ` catch-all that was silently dropping every message
SYSTEM_PATTERNS = re.compile(
    r"^(Messages and calls are end-to-end|"
    r"Your security code|"
    r".+ added .+|.+ left|.+ removed .+|"
    r".+ changed the|.+ created|"
    r".+ was added|.+ joined|"
    r".+ changed their|You were added|"
    r"This message was deleted|null|<null>|"
    r"Missed voice call|Missed video call|"
    r".+ changed this group)",
    re.IGNORECASE,
)

# FIX: Each pattern compiled separately — avoids combined-regex group index drift
_PATTERNS = [re.compile(p) for p in CHAT_PATTERNS]

_MEDIA_RE = re.compile(
    r"<Media omitted>|image omitted|video omitted|sticker omitted|"
    r"audio omitted|document omitted|GIF omitted|<image omitted>|"
    r"<video omitted>|<audio omitted>|<document omitted>",
    flags=re.IGNORECASE,
)
_DELETED_RE = re.compile(
    r"This message was deleted|You deleted this message|"
    r"message was deleted|deleted message",
    flags=re.IGNORECASE,
)
_EDITED_RE       = re.compile(r"<This message was edited>|\(edited\)", flags=re.IGNORECASE)
_FORWARDED_RE    = re.compile(r"Forwarded|‎Forwarded", flags=re.IGNORECASE)
_URL_RE          = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE        = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)
_MENTION_RE      = re.compile(r"@[\w\d]+")
_TOXIC_WORDS_RE  = re.compile(
    r"\b(hate|kill|stupid|idiot|dumb|ugly|useless|moron|loser|"
    r"shut up|go away|worthless|pathetic|disgusting|horrible)\b",
    flags=re.IGNORECASE,
)

# Colour palette
C_INDIGO  = "#6366f1"
C_VIOLET  = "#8b5cf6"
C_CYAN    = "#06b6d4"
C_GREEN   = "#10b981"
C_AMBER   = "#f59e0b"
C_RED     = "#ef4444"
C_PINK    = "#ec4899"
C_SLATE   = "#94a3b8"
C_TEAL    = "#14b8a6"
C_ORANGE  = "#f97316"


# ══════════════════════════════════════════════════════════════════════════════
# PARSING LAYER
# ══════════════════════════════════════════════════════════════════════════════

def decode_bytes(raw: bytes) -> str:
    """Decode raw file bytes trying multiple encodings."""
    for enc in ENCODING_FALLBACKS:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(
        f"Could not decode with any of: {', '.join(ENCODING_FALLBACKS)}. "
        "Re-export the chat as UTF-8."
    )


def parse_chat(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse a WhatsApp .txt export into a tidy DataFrame.

    Handles all known regional timestamp formats, multi-line messages,
    system notifications, deleted/edited/forwarded message detection.

    Returns
    -------
    pd.DataFrame with columns: timestamp, author, message,
        is_deleted, is_edited, is_forwarded.
    """
    text = decode_bytes(file_bytes)
    rows: list = []
    current: Optional[dict] = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # FIX: Skip system/event lines by matching against the full line
        if SYSTEM_PATTERNS.match(line):
            continue

        # FIX: Try each compiled pattern individually — avoids group-index drift
        m = None
        for pat in _PATTERNS:
            m = pat.match(line)
            if m:
                break

        if m:
            g = m.groups()
            # Every pattern yields exactly 4 groups: date, time, author, message
            date_str  = g[0].strip()
            time_str  = g[1].strip()
            author    = g[2].strip()
            message   = g[3].strip()
            timestamp = f"{date_str} {time_str}"

            if current:
                rows.append(current)

            current = {
                "timestamp":    timestamp,
                "author":       author,
                "message":      message,
                "is_deleted":   bool(_DELETED_RE.search(message)),
                "is_edited":    bool(_EDITED_RE.search(message)),
                "is_forwarded": bool(_FORWARDED_RE.search(message)),
            }
        elif current:
            # Continuation line (multi-line message)
            current["message"] += f" {line}"

    if current:
        rows.append(current)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "author", "message",
                                     "is_deleted", "is_edited", "is_forwarded"])

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING / ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_ts(series: pd.Series) -> pd.Series:
    """
    Normalise timestamp strings before format matching.
    - Replace Narrow No-Break Space U+202F (used by WhatsApp between time and am/pm)
    - Upper-case am/pm so strptime %p matches on all platforms
    """
    return (
        series
        .str.replace("\u202f", " ", regex=False)   # narrow nbsp → space
        .str.replace(r"\bam\b", "AM", regex=True)  # lowercase am → AM
        .str.replace(r"\bpm\b", "PM", regex=True)  # lowercase pm → PM
    )


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Try known date formats then fall back to pandas inference."""
    normalised = _normalise_ts(series)
    for fmt in DATETIME_FORMATS:
        try:
            parsed = pd.to_datetime(normalised, format=fmt, errors="raise")
            if parsed.notna().sum() > len(normalised) * 0.8:
                return parsed
        except (ValueError, TypeError):
            continue
    logger.warning("Falling back to inferred timestamp parsing.")
    return pd.to_datetime(normalised, errors="coerce")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns to the raw parsed DataFrame."""
    if df.empty:
        return df

    df = df.copy()

    # Timestamps
    df["datetime"] = _parse_timestamps(df["timestamp"])
    df.dropna(subset=["datetime"], inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    dt              = df["datetime"].dt
    df["date"]      = dt.date
    df["hour"]      = dt.hour
    df["minute"]    = dt.minute
    df["day_name"]  = dt.day_name()
    df["month"]     = dt.to_period("M").astype(str)
    df["week"]      = dt.to_period("W").astype(str)
    df["quarter"]   = dt.to_period("Q").astype(str)
    df["year"]      = dt.year
    df["is_weekend"] = dt.dayofweek >= 5

    # Boolean flags
    df["is_media"]    = df["message"].apply(lambda x: bool(_MEDIA_RE.search(x)))
    df["has_url"]     = df["message"].apply(lambda x: bool(_URL_RE.search(x)))
    df["has_mention"] = df["message"].apply(lambda x: bool(_MENTION_RE.search(x)))
    df["is_toxic"]    = df["message"].apply(lambda x: bool(_TOXIC_WORDS_RE.search(x)))

    # Emoji extraction
    df["emojis"]      = df["message"].apply(lambda x: _EMOJI_RE.findall(x))
    df["emoji_count"] = df["emojis"].apply(len)

    # Word / char counts
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))
    df["char_count"] = df["message"].apply(len)

    # Cleaned text
    df["message_clean"] = (
        df["message"]
        .str.replace(_URL_RE, " ", regex=True)
        .str.replace(_MEDIA_RE, " ", regex=True)
        .str.replace(_DELETED_RE, " ", regex=True)
        .str.replace(_EDITED_RE, " ", regex=True)
        .str.replace(_EMOJI_RE, " ", regex=True)
        .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        .str.lower()
        .str.strip()
    )

    # Question detection
    df["is_question"] = df["message"].str.contains(r"\?", regex=False)

    # Response time
    df["response_gap_min"] = (
        df["datetime"].diff().dt.total_seconds() / 60
    ).where(
        (df["author"] != df["author"].shift(1)) &
        (df["datetime"].diff().dt.total_seconds() / 60 < 1440),
        other=np.nan,
    )

    # Conversation session (gap > 1hr = new session)
    gap_hrs = df["datetime"].diff().dt.total_seconds() / 3600
    df["session_id"] = (gap_hrs > 1).cumsum()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS LAYER
# ══════════════════════════════════════════════════════════════════════════════

def overview_stats(df: pd.DataFrame) -> dict:
    """Compute top-level KPI statistics."""
    if df.empty:
        return {}
    nm = df[~df["is_media"]]
    total_days = max((df["datetime"].max() - df["datetime"].min()).days, 1)
    return {
        "total_messages":        len(df),
        "total_words":           int(nm["word_count"].sum()),
        "media_shared":          int(df["is_media"].sum()),
        "links_shared":          int(df["has_url"].sum()),
        "emojis_sent":           int(df["emoji_count"].sum()),
        "avg_words_per_message": round(nm["word_count"].mean(), 1) if len(nm) else 0,
        "unique_authors":        df["author"].nunique(),
        "date_range_days":       total_days,
        "avg_msgs_per_day":      round(len(df) / total_days, 1),
        "total_sessions":        int(df["session_id"].nunique()),
        "deleted_messages":      int(df["is_deleted"].sum()),
        "edited_messages":       int(df["is_edited"].sum()),
        "forwarded_messages":    int(df["is_forwarded"].sum()),
        "questions_asked":       int(df["is_question"].sum()),
        "toxic_messages":        int(df["is_toxic"].sum()),
        "first_date":            df["datetime"].min().strftime("%b %d, %Y"),
        "last_date":             df["datetime"].max().strftime("%b %d, %Y"),
    }


def author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-author aggregated statistics."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("author", sort=False).agg(
        Messages     = ("message", "count"),
        Words        = ("word_count", "sum"),
        Characters   = ("char_count", "sum"),
        Media        = ("is_media", "sum"),
        Links        = ("has_url", "sum"),
        Emojis       = ("emoji_count", "sum"),
        Questions    = ("is_question", "sum"),
        Deleted      = ("is_deleted", "sum"),
        Edited       = ("is_edited", "sum"),
        Forwarded    = ("is_forwarded", "sum"),
    ).reset_index().rename(columns={"author": "Author"})

    agg["Avg Words/Msg"]      = (agg["Words"] / agg["Messages"]).round(1)
    agg["% of Total"]         = (agg["Messages"] / agg["Messages"].sum() * 100).round(1)
    agg["Emojis/Msg"]         = (agg["Emojis"] / agg["Messages"]).round(2)
    agg["Media %"]            = (agg["Media"] / agg["Messages"] * 100).round(1)

    # Avg response time per author
    rt = df[df["response_gap_min"].notna()].groupby("author")["response_gap_min"].mean().round(1)
    agg["Avg Response (min)"] = agg["Author"].map(rt).fillna(0)

    # Weekend activity ratio
    wa = df.groupby("author").apply(
        lambda x: (x["is_weekend"].sum() / len(x) * 100)
    ).round(1)
    agg["Weekend %"] = agg["Author"].map(wa).fillna(0)

    return agg.sort_values("Messages", ascending=False).reset_index(drop=True)


def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Message counts per calendar date."""
    return (
        df.groupby("date").size()
        .reset_index(name="count")
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )


def hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Message counts per hour (0-23)."""
    h = df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    h.columns = ["hour", "count"]
    return h


def day_of_week_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Message counts per weekday."""
    c = df.groupby("day_name").size().reindex(WEEKDAY_ORDER, fill_value=0).reset_index()
    c.columns = ["day", "count"]
    return c


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Message counts per calendar month."""
    return df.groupby("month").size().reset_index(name="count").sort_values("month")


def heatmap_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week × hour-of-day pivot for the activity heatmap."""
    return (
        df.groupby(["day_name", "hour"]).size()
        .unstack(fill_value=0)
        .reindex(WEEKDAY_ORDER, fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )


def word_frequency(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Word frequencies excluding stop-words and tokens < 3 chars."""
    combined = " ".join(df["message_clean"].dropna().astype(str)).lower()
    tokens   = re.findall(r"\b[a-zA-Z]{3,}\b", combined)
    filtered = [w for w in tokens if w not in STOP_WORDS]
    freq     = Counter(filtered).most_common(top_n)
    return pd.DataFrame(freq, columns=["word", "count"])


def emoji_frequency(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Emoji usage frequency."""
    all_emojis = [e for row in df["emojis"] for e in row]
    freq = Counter(all_emojis).most_common(top_n)
    return pd.DataFrame(freq, columns=["emoji", "count"])


def longest_messages(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Longest non-media, non-deleted messages."""
    nm = df[~df["is_media"] & ~df["is_deleted"]].copy()
    result = nm.nlargest(top_n, "word_count")[["author", "date", "message", "word_count"]]
    return result.rename(columns={
        "author": "Author", "date": "Date",
        "message": "Message", "word_count": "Words",
    }).reset_index(drop=True)


def most_active_days(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top-N dates by message volume."""
    return (
        df.groupby("date").size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="Messages")
        .rename(columns={"date": "Date"})
    )


def response_times_df(df: pd.DataFrame) -> pd.DataFrame:
    """Average reply latency per author (minutes)."""
    if df.empty or df["author"].nunique() < 2:
        return pd.DataFrame()
    valid = df[df["response_gap_min"].notna() & (df["response_gap_min"] > 0)]
    if valid.empty:
        return pd.DataFrame()
    result = (
        valid.groupby("author")["response_gap_min"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={
            "author": "Author",
            "mean":   "Avg Response (min)",
            "median": "Median Response (min)",
            "count":  "Responses",
        })
    )
    result["Avg Response (min)"]    = result["Avg Response (min)"].round(1)
    result["Median Response (min)"] = result["Median Response (min)"].round(1)
    return result.sort_values("Avg Response (min)").reset_index(drop=True)


def conversation_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Find longest consecutive day streaks per author."""
    results = []
    for author, grp in df.groupby("author"):
        dates = sorted(grp["date"].unique())
        max_streak, cur_streak = 1, 1
        for i in range(1, len(dates)):
            diff = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i-1])).days
            cur_streak = cur_streak + 1 if diff == 1 else 1
            max_streak = max(max_streak, cur_streak)
        results.append({"Author": author, "Longest Streak (days)": max_streak, "Active Days": len(dates)})
    return pd.DataFrame(results).sort_values("Longest Streak (days)", ascending=False).reset_index(drop=True)


def get_ngrams(df: pd.DataFrame, n: int = 2, top_k: int = 20) -> pd.DataFrame:
    """Compute top N-grams from cleaned messages."""
    texts = df["message_clean"].dropna().astype(str)
    all_tokens = []
    for text in texts:
        tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
        if len(tokens) >= n:
            all_tokens.extend([" ".join(g) for g in zip(*[tokens[i:] for i in range(n)])])
    freq = Counter(all_tokens).most_common(top_k)
    return pd.DataFrame(freq, columns=["ngram", "count"])


def tfidf_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Extract top TF-IDF keywords from the corpus."""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    texts = df["message_clean"].dropna().astype(str).tolist()
    texts = [t for t in texts if len(t.split()) > 2]
    if len(texts) < 5:
        return pd.DataFrame()
    try:
        vec = TfidfVectorizer(
            max_features=500, stop_words="english",
            ngram_range=(1, 2), min_df=2,
        )
        X = vec.fit_transform(texts)
        scores = np.array(X.mean(axis=0)).flatten()
        terms  = vec.get_feature_names_out()
        idx    = np.argsort(scores)[::-1][:top_n]
        return pd.DataFrame({"keyword": terms[idx], "score": scores[idx].round(4)})
    except Exception:
        return pd.DataFrame()


def topic_modeling(df: pd.DataFrame, n_topics: int = 5, top_words: int = 8) -> list:
    """LDA topic modeling on message corpus."""
    if not SKLEARN_AVAILABLE:
        return []
    texts = df["message_clean"].dropna().astype(str).tolist()
    texts = [t for t in texts if len(t.split()) > 3]
    if len(texts) < 20:
        return []
    try:
        vec = TfidfVectorizer(
            max_features=200, stop_words="english",
            min_df=3, max_df=0.85,
        )
        X = vec.fit_transform(texts)
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42,
            max_iter=10, learning_method="batch",
        )
        lda.fit(X)
        terms = vec.get_feature_names_out()
        topics = []
        for i, comp in enumerate(lda.components_):
            top_idx   = comp.argsort()[::-1][:top_words]
            top_terms = [terms[j] for j in top_idx]
            topics.append({"topic": i + 1, "words": top_terms, "weight": float(comp[top_idx].mean())})
        return topics
    except Exception:
        return []


def cluster_users(df: pd.DataFrame) -> pd.DataFrame:
    """K-Means clustering of users by chat behaviour."""
    if not SKLEARN_AVAILABLE or df["author"].nunique() < 3:
        return pd.DataFrame()
    try:
        a = author_stats(df)[["Author", "Messages", "Avg Words/Msg", "Emojis/Msg",
                               "Media %", "Avg Response (min)", "Weekend %"]].copy()
        feat = a.drop("Author", axis=1).fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(feat)
        k = min(3, len(a))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        a["Cluster"] = km.fit_predict(X)
        cluster_names = {0: "🗣 Talkers", 1: "📸 Media Sharers", 2: "⚡ Quick Responders"}
        a["Profile"] = a["Cluster"].map(cluster_names)
        return a[["Author", "Messages", "Avg Words/Msg", "Profile"]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest anomaly detection on daily message counts."""
    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()
    try:
        daily = daily_counts(df)
        if len(daily) < 10:
            return pd.DataFrame()
        X = daily[["count"]].values
        iso = IsolationForest(contamination=0.05, random_state=42)
        daily["anomaly"] = iso.fit_predict(X)
        anomalies = daily[daily["anomaly"] == -1][["date", "count"]].copy()
        anomalies.columns = ["Date", "Messages"]
        return anomalies.sort_values("Messages", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def forecast_messages(df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
    """Simple exponential smoothing forecast for next N days."""
    try:
        daily = daily_counts(df).set_index("date")["count"]
        if len(daily) < 14:
            return pd.DataFrame()
        alpha = 0.3
        smoothed = daily.ewm(alpha=alpha, adjust=False).mean()
        last_smooth = smoothed.iloc[-1]
        last_date   = daily.index[-1]
        future = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq="D")
        recent = daily.tail(28)
        dow_avg = recent.groupby(recent.index.dayofweek).mean()
        overall_avg = recent.mean()
        preds = []
        for d in future:
            dow_factor = dow_avg.get(d.dayofweek, overall_avg) / overall_avg if overall_avg > 0 else 1.0
            preds.append(round(max(0, float(last_smooth * dow_factor)), 1))
        return pd.DataFrame({"date": future, "forecast": preds})
    except Exception:
        return pd.DataFrame()


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """VADER sentiment annotation per message."""
    if not VADER_AVAILABLE or df.empty:
        return df
    sia    = SentimentIntensityAnalyzer()
    scores = df["message_clean"].apply(lambda t: sia.polarity_scores(str(t)))
    sdf    = pd.DataFrame(scores.tolist())
    sdf.columns = ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]
    sdf["sentiment_label"] = sdf["sentiment_compound"].apply(
        lambda v: "Positive" if v >= 0.05 else ("Negative" if v <= -0.05 else "Neutral")
    )
    return pd.concat([df.reset_index(drop=True), sdf], axis=1)


def author_sentiment(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-author mean sentiment statistics."""
    if "sentiment_compound" not in df.columns:
        return None
    agg = df.groupby("author").agg(
        avg=("sentiment_compound", "mean"),
        pos=("sentiment_label", lambda x: (x == "Positive").mean() * 100),
        neg=("sentiment_label", lambda x: (x == "Negative").mean() * 100),
        neu=("sentiment_label", lambda x: (x == "Neutral").mean() * 100),
    ).reset_index()
    agg.columns = ["Author", "Avg Sentiment", "Positive %", "Negative %", "Neutral %"]
    for col in ["Avg Sentiment", "Positive %", "Negative %", "Neutral %"]:
        agg[col] = agg[col].round(2)
    return agg.sort_values("Avg Sentiment", ascending=False).reset_index(drop=True)


def detect_language(df: pd.DataFrame) -> pd.DataFrame:
    """Detect language for each message (sample for performance)."""
    if not LANGDETECT_AVAILABLE:
        return pd.DataFrame()
    texts = df["message_clean"].dropna().astype(str)
    texts = texts[texts.str.len() > 10]
    if texts.empty:
        return pd.DataFrame()
    sample_size = min(200, len(texts))
    texts = texts.sample(sample_size, random_state=42)
    langs = []
    for t in texts:
        try:
            langs.append(lang_detect(t))
        except Exception:
            langs.append("unknown")
    lang_counts = Counter(langs)
    return pd.DataFrame(lang_counts.most_common(10), columns=["Language", "Count"])


def generate_wordcloud(df: pd.DataFrame, author: Optional[str] = None) -> Optional[bytes]:
    """Generate a word cloud image and return as PNG bytes."""
    if not WORDCLOUD_AVAILABLE:
        return None
    text = " ".join(df["message_clean"].dropna().astype(str).tolist())
    text = " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)
    if not text.strip():
        return None
    try:
        wc = WordCloud(
            width=900, height=450,
            background_color="#0f0f1a",
            colormap="cool",
            max_words=150,
            prefer_horizontal=0.8,
            collocations=False,
            min_font_size=10,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0f0f1a")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f0f1a", dpi=150)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None


def auto_insights(df: pd.DataFrame, ov: dict) -> list:
    """Generate human-readable insight strings from analytics."""
    insights = []
    a_stats  = author_stats(df)

    if not a_stats.empty:
        top_author = a_stats.iloc[0]
        insights.append(
            f"🏆 **{top_author['Author']}** is the most active member with "
            f"**{int(top_author['Messages']):,}** messages ({top_author['% of Total']}% of total)."
        )
        if len(a_stats) > 1:
            fastest = a_stats.loc[a_stats["Avg Response (min)"] > 0, "Avg Response (min)"]
            if not fastest.empty:
                idx = fastest.idxmin()
                insights.append(
                    f"⚡ **{a_stats.loc[idx, 'Author']}** responds fastest on average "
                    f"({a_stats.loc[idx, 'Avg Response (min)']} min)."
                )
        emoji_king = a_stats.loc[a_stats["Emojis/Msg"].idxmax()]
        if emoji_king["Emojis/Msg"] > 0:
            insights.append(
                f"😄 **{emoji_king['Author']}** uses the most emojis per message "
                f"({emoji_king['Emojis/Msg']:.2f} emojis/msg)."
            )

    dow = day_of_week_counts(df)
    if not dow.empty:
        peak_day = dow.loc[dow["count"].idxmax(), "day"]
        insights.append(f"📅 The group is most active on **{peak_day}**.")

    hourly = hourly_counts(df)
    if not hourly.empty:
        peak_h = int(hourly.loc[hourly["count"].idxmax(), "hour"])
        suffix = "AM" if peak_h < 12 else "PM"
        h12    = peak_h if peak_h <= 12 else peak_h - 12
        h12    = 12 if h12 == 0 else h12
        insights.append(f"⏰ Chat peaks at **{h12} {suffix}** daily.")

    if ov.get("deleted_messages", 0) > 0:
        insights.append(f"🗑 **{ov['deleted_messages']}** messages were deleted.")
    if ov.get("forwarded_messages", 0) > 0:
        insights.append(f"↪️ **{ov['forwarded_messages']}** messages were forwarded.")
    if ov.get("toxic_messages", 0) > 0:
        insights.append(
            f"⚠️ **{ov['toxic_messages']}** messages contain potentially toxic language."
        )

    daily = daily_counts(df)
    if not daily.empty:
        busiest = daily.loc[daily["count"].idxmax()]
        insights.append(
            f"🔥 Busiest day was **{pd.Timestamp(busiest['date']).strftime('%b %d, %Y')}** "
            f"with **{int(busiest['count'])}** messages."
        )

    insights.append(
        f"📊 The chat spans **{ov.get('date_range_days', 0)}** days across "
        f"**{ov.get('total_sessions', 0)}** conversation sessions."
    )
    return insights


# ══════════════════════════════════════════════════════════════════════════════
# CHART FACTORY
# ══════════════════════════════════════════════════════════════════════════════

_BASE = dict(
    template=PLOTLY_TPL,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=12, r=12, t=48, b=12),
    font=dict(family="'Sora', 'DM Sans', sans-serif", size=12, color="#cbd5e1"),
)


def _fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#e2e8f0")), **_BASE)
    return fig


def chart_timeline(daily: pd.DataFrame, forecast: pd.DataFrame = None) -> go.Figure:
    """Line + area chart with optional forecast."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["count"],
        mode="lines", name="Daily",
        line=dict(color=C_INDIGO, width=1.5),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
    ))
    daily["roll7"] = daily["count"].rolling(7, min_periods=1).mean().round(1)
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["roll7"],
        mode="lines", name="7-day avg",
        line=dict(color=C_CYAN, width=2, dash="dot"),
    ))
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color=C_AMBER, width=2, dash="dash"),
            marker=dict(size=5),
        ))
        fig.add_vrect(
            x0=forecast["date"].iloc[0], x1=forecast["date"].iloc[-1],
            fillcolor="rgba(245,158,11,0.05)", line_width=0,
        )
    fig.update_layout(
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        xaxis_title="Date", yaxis_title="Messages",
    )
    return _fig(fig, "📅 Daily Message Activity + Forecast")


def chart_hourly(hourly: pd.DataFrame) -> go.Figure:
    fig = px.bar(hourly, x="hour", y="count",
                 color="count", color_continuous_scale="Blues",
                 labels={"hour": "Hour of Day", "count": "Messages"})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, xaxis=dict(tickmode="linear"))
    return _fig(fig, "⏰ Activity by Hour")


def chart_dow(dow: pd.DataFrame) -> go.Figure:
    fig = px.bar(dow[::-1], x="count", y="day", orientation="h",
                 color="count", color_continuous_scale="Teal",
                 labels={"day": "", "count": "Messages"})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False)
    return _fig(fig, "📆 Activity by Day of Week")


def chart_monthly(monthly: pd.DataFrame) -> go.Figure:
    fig = px.bar(monthly, x="month", y="count",
                 color="count", color_continuous_scale="Purples",
                 labels={"month": "Month", "count": "Messages"})
    fig.update_traces(marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
    return _fig(fig, "📊 Monthly Trends")


def chart_heatmap(pivot: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day", color="Messages"),
        x=[str(h) for h in range(24)],
        y=list(pivot.index),
        color_continuous_scale="Inferno",
        aspect="auto",
    )
    fig.update_traces(hoverongaps=False)
    fig.update_layout(
        xaxis_nticks=24,
        coloraxis_colorbar=dict(title="Msgs", thickness=10),
    )
    return _fig(fig, "🔥 Activity Heatmap (Day × Hour)")


def chart_author_pie(a_stats: pd.DataFrame) -> go.Figure:
    fig = px.pie(a_stats, values="Messages", names="Author",
                 hole=0.45, color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=False)
    return _fig(fig, "👥 Message Share")


def chart_author_metric(a_stats: pd.DataFrame, metric: str, scale: str) -> go.Figure:
    fig = px.bar(
        a_stats.sort_values(metric),
        x=metric, y="Author", orientation="h",
        color=metric, color_continuous_scale=scale,
        text=metric,
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return _fig(fig, f"📝 {metric} per Author")


def chart_words(word_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        word_df.sort_values("count"),
        x="count", y="word", orientation="h",
        color="count", color_continuous_scale="Viridis",
        labels={"count": "Frequency", "word": "Word"},
        text="count",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return _fig(fig, "💬 Top Words")


def chart_ngrams(ngram_df: pd.DataFrame, n: int) -> go.Figure:
    label = "Bigrams" if n == 2 else "Trigrams"
    fig = px.bar(
        ngram_df.sort_values("count"),
        x="count", y="ngram", orientation="h",
        color="count", color_continuous_scale="Tealgrn",
        text="count",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return _fig(fig, f"🔗 Top {label}")


def chart_tfidf(kw_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        kw_df.sort_values("score"),
        x="score", y="keyword", orientation="h",
        color="score", color_continuous_scale="Plasma",
        text="score",
    )
    fig.update_traces(textposition="outside", marker_line_width=0, texttemplate="%{text:.3f}")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="")
    return _fig(fig, "🔑 TF-IDF Keywords")


def chart_emojis(emoji_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        emoji_df,
        x="emoji", y="count",
        color="count", color_continuous_scale="Oranges",
        text="count",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="")
    return _fig(fig, "😄 Most Used Emojis")


def chart_response_times(rt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rt["Author"], y=rt["Avg Response (min)"],
        name="Avg", marker_color=C_INDIGO,
    ))
    fig.add_trace(go.Bar(
        x=rt["Author"], y=rt["Median Response (min)"],
        name="Median", marker_color=C_CYAN,
    ))
    fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Minutes")
    return _fig(fig, "⚡ Response Time Comparison")


def chart_sentiment_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    cmap = {"Positive": C_GREEN, "Neutral": C_INDIGO, "Negative": C_RED}
    fig = px.pie(counts, values="Count", names="Label", hole=0.45,
                 color="Label", color_discrete_map=cmap)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=False)
    return _fig(fig, "😊 Overall Sentiment")


def chart_sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    daily = (
        df.groupby("date")["sentiment_compound"].mean()
        .reset_index().sort_values("date")
    )
    daily["date"]    = pd.to_datetime(daily["date"])
    daily["rolling"] = daily["sentiment_compound"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    pos_mask = daily["rolling"] >= 0
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling"].where(pos_mask, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(16,185,129,0.15)", name="Positive",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling"].where(~pos_mask, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(239,68,68,0.15)", name="Negative",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling"],
        mode="lines", line=dict(color=C_CYAN, width=2), name="Avg",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(200,200,200,0.3)")
    fig.update_layout(yaxis_title="Compound score", xaxis_title="Date")
    return _fig(fig, "📈 Sentiment Over Time")


def chart_author_sentiment_radar(as_df: pd.DataFrame) -> go.Figure:
    """Radar chart of per-author sentiment split."""
    fig = go.Figure()
    categories = ["Positive %", "Neutral %", "Negative %"]
    for _, row in as_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row["Positive %"], row["Neutral %"], row["Negative %"]],
            theta=categories,
            fill="toself",
            name=row["Author"],
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return _fig(fig, "🎯 Per-Author Sentiment Radar")


def chart_conversation_flow(df: pd.DataFrame) -> go.Figure:
    """Sankey-style conversation flow between authors."""
    if df["author"].nunique() < 2:
        return go.Figure()
    authors = df["author"].tolist()
    pairs = Counter(zip(authors[:-1], authors[1:]))
    top_pairs = [(k, v) for k, v in pairs.most_common(20) if k[0] != k[1]]
    if not top_pairs:
        return go.Figure()
    all_nodes = list(dict.fromkeys([a for p, _ in top_pairs for a in p]))
    node_idx  = {n: i for i, n in enumerate(all_nodes)}
    sources   = [node_idx[p[0]] for p, _ in top_pairs]
    targets   = [node_idx[p[1]] for p, _ in top_pairs]
    values    = [v for _, v in top_pairs]
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=all_nodes,
            color=[C_INDIGO, C_CYAN, C_VIOLET, C_TEAL, C_PINK][:len(all_nodes)],
        ),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(99,102,241,0.2)"),
    ))
    return _fig(fig, "🔄 Conversation Flow")


def chart_anomalies(daily: pd.DataFrame, anomalies: pd.DataFrame) -> go.Figure:
    """Timeline with anomaly markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["count"],
        mode="lines", name="Daily",
        line=dict(color=C_SLATE, width=1),
        fill="tozeroy", fillcolor="rgba(148,163,184,0.05)",
    ))
    if not anomalies.empty:
        anom_merged = daily[daily["date"].isin(pd.to_datetime(anomalies["Date"]))]
        fig.add_trace(go.Scatter(
            x=anom_merged["date"], y=anom_merged["count"],
            mode="markers", name="Anomaly",
            marker=dict(color=C_RED, size=10, symbol="x"),
        ))
    return _fig(fig, "⚠️ Activity Anomalies")


def chart_cluster_scatter(cluster_df: pd.DataFrame) -> go.Figure:
    """Scatter of users by message count and avg words, colored by cluster."""
    if cluster_df.empty:
        return go.Figure()
    fig = px.scatter(
        cluster_df, x="Messages", y="Avg Words/Msg", text="Author",
        color="Profile", size="Messages",
        color_discrete_sequence=[C_INDIGO, C_CYAN, C_AMBER],
    )
    fig.update_traces(textposition="top center")
    return _fig(fig, "🧩 User Behaviour Clusters")


def chart_message_types(df: pd.DataFrame) -> go.Figure:
    """Stacked bar of message types per author."""
    a = df.groupby("author").agg(
        Text   = ("is_media", lambda x: (~x).sum()),
        Media  = ("is_media", "sum"),
        Links  = ("has_url", "sum"),
    ).reset_index()
    fig = px.bar(a, x="author", y=["Text", "Media", "Links"],
                 barmode="stack",
                 color_discrete_sequence=[C_INDIGO, C_CYAN, C_AMBER])
    fig.update_layout(xaxis_title="", yaxis_title="Count",
                      legend_title="Type")
    return _fig(fig, "📦 Message Types per Author")


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def kpi_card(col, icon: str, label: str, value: str, delta: str = "") -> None:
    col.metric(f"{icon} {label}", value, delta=delta if delta else None)


def render_kpi_strip(ov: dict) -> None:
    c = st.columns(6)
    items = [
        ("💬", "Messages",      f"{ov.get('total_messages', 0):,}"),
        ("📝", "Words",         f"{ov.get('total_words', 0):,}"),
        ("📸", "Media",         f"{ov.get('media_shared', 0):,}"),
        ("🔗", "Links",         f"{ov.get('links_shared', 0):,}"),
        ("😄", "Emojis",        f"{ov.get('emojis_sent', 0):,}"),
        ("⚡", "Avg Words/Msg", str(ov.get("avg_words_per_message", 0))),
    ]
    for col, (icon, label, val) in zip(c, items):
        col.metric(f"{icon} {label}", val)


def render_kpi_strip_2(ov: dict) -> None:
    c = st.columns(6)
    items = [
        ("🗑", "Deleted",    f"{ov.get('deleted_messages', 0):,}"),
        ("✏️", "Edited",     f"{ov.get('edited_messages', 0):,}"),
        ("↪️", "Forwarded", f"{ov.get('forwarded_messages', 0):,}"),
        ("❓", "Questions",  f"{ov.get('questions_asked', 0):,}"),
        ("🗓", "Sessions",   f"{ov.get('total_sessions', 0):,}"),
        ("📅", "Days",       f"{ov.get('date_range_days', 0):,}"),
    ]
    for col, (icon, label, val) in zip(c, items):
        col.metric(f"{icon} {label}", val)


def section(title: str) -> None:
    st.markdown(
        f'<p style="font-size:1.1rem;font-weight:700;color:#e2e8f0;'
        f'border-left:4px solid {C_INDIGO};padding-left:10px;margin:20px 0 8px">'
        f'{title}</p>',
        unsafe_allow_html=True,
    )
    st.divider()


def table(df: pd.DataFrame, filename: str = "data.csv") -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download CSV",
        data=df.to_csv(index=False).encode(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def upload_help() -> None:
    with st.expander("📖 How to export your WhatsApp chat", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Android**
1. Open the chat in WhatsApp
2. Tap **⋮** → **More** → **Export chat**
3. Choose **Without Media**
4. Share the `.txt` file to your device
""")
        with col2:
            st.markdown("""
**iPhone**
1. Open the chat in WhatsApp
2. Tap the contact / group name
3. Scroll down → **Export Chat**
4. Choose **Without Media**
5. Save the `.txt` file
""")


def search_messages(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Full-text search across messages."""
    if not query.strip():
        return pd.DataFrame()
    mask = df["message"].str.contains(query, case=False, na=False, regex=False)
    result = df[mask][["datetime", "author", "message"]].copy()
    result.columns = ["Timestamp", "Author", "Message"]
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def tab_overview(df: pd.DataFrame, ov: dict) -> None:
    section("🔍 Auto Insights")
    insights = auto_insights(df, ov)
    cols = st.columns(2)
    for i, insight in enumerate(insights):
        cols[i % 2].info(insight)

    section("📊 Activity Overview")
    daily = daily_counts(df)
    fc    = forecast_messages(df, days_ahead=7)
    st.plotly_chart(chart_timeline(daily, fc), use_container_width=True, key="overview_timeline")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_message_types(df), use_container_width=True, key="overview_msg_types")
    with c2:
        st.plotly_chart(chart_author_pie(author_stats(df)), use_container_width=True, key="overview_author_pie")

    section("🗓 Most Active Days")
    table(most_active_days(df), "most_active_days.csv")


def tab_activity(df: pd.DataFrame) -> None:
    section("Activity Patterns")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_hourly(hourly_counts(df)), use_container_width=True, key="act_hourly")
    with c2:
        st.plotly_chart(chart_dow(day_of_week_counts(df)), use_container_width=True, key="act_dow")

    st.plotly_chart(chart_monthly(monthly_counts(df)), use_container_width=True, key="act_monthly")

    section("🔥 Heatmap")
    pivot = heatmap_pivot(df)
    if not pivot.empty:
        st.plotly_chart(chart_heatmap(pivot), use_container_width=True, key="act_heatmap")

    section("⚠️ Anomaly Detection")
    if SKLEARN_AVAILABLE:
        daily = daily_counts(df)
        anomalies = detect_anomalies(df)
        st.plotly_chart(chart_anomalies(daily, anomalies), use_container_width=True, key="act_anomalies")
        if not anomalies.empty:
            st.caption(f"Found **{len(anomalies)}** anomalous days (unusually high/low activity).")
            table(anomalies, "anomalies.csv")
        else:
            st.success("No significant activity anomalies detected.", icon="✅")
    else:
        st.info("Install `scikit-learn` for anomaly detection.", icon="💡")


def tab_authors(df: pd.DataFrame, selected: str) -> None:
    section("Author Breakdown")

    a = author_stats(df)
    if a.empty:
        st.info("No author data available.", icon="ℹ️")
        return

    if selected != "All":
        a_row = a[a["Author"] == selected]
        if not a_row.empty:
            row = a_row.iloc[0]
            c = st.columns(4)
            c[0].metric("💬 Messages", f"{int(row['Messages']):,}")
            c[1].metric("📝 Avg Words", f"{row['Avg Words/Msg']}")
            c[2].metric("😄 Emojis/Msg", f"{row['Emojis/Msg']}")
            c[3].metric("⚡ Avg Response", f"{row['Avg Response (min)']} min")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_author_pie(a), use_container_width=True, key="auth_pie")
        with c2:
            st.plotly_chart(chart_author_metric(a, "Words", "Purples"), use_container_width=True, key="auth_words")

        st.plotly_chart(chart_author_metric(a, "Emojis", "Oranges"), use_container_width=True, key="auth_emojis")
        st.plotly_chart(chart_author_metric(a, "Media %", "Reds"), use_container_width=True, key="auth_media")

    section("📊 Full Stats Table")
    display_cols = ["Author", "Messages", "Words", "Avg Words/Msg", "% of Total",
                    "Emojis/Msg", "Media %", "Questions", "Avg Response (min)", "Weekend %"]
    table(a[[c for c in display_cols if c in a.columns]], "author_stats.csv")

    section("⚡ Response Times")
    rt = response_times_df(df)
    if not rt.empty:
        st.plotly_chart(chart_response_times(rt), use_container_width=True, key="auth_response_times")
        table(rt, "response_times.csv")

    section("🏆 Conversation Streaks")
    streaks = conversation_streaks(df)
    table(streaks, "streaks.csv")

    section("🔄 Conversation Flow")
    st.plotly_chart(chart_conversation_flow(df), use_container_width=True, key="auth_flow")

    if SKLEARN_AVAILABLE and df["author"].nunique() >= 3:
        section("🧩 User Behaviour Clusters")
        cluster_df = cluster_users(df)
        if not cluster_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(chart_cluster_scatter(cluster_df), use_container_width=True, key="auth_cluster")
            with c2:
                table(cluster_df, "clusters.csv")


def tab_content(df: pd.DataFrame) -> None:
    section("Word Frequency")
    wf = word_frequency(df, top_n=30)
    if not wf.empty:
        st.plotly_chart(chart_words(wf), use_container_width=True, key="content_words")

    if WORDCLOUD_AVAILABLE:
        section("☁️ Word Cloud")
        wc_bytes = generate_wordcloud(df)
        if wc_bytes:
            st.image(wc_bytes, use_container_width=True)
        else:
            st.info("Not enough text for a word cloud.", icon="ℹ️")
    else:
        st.info("Install `wordcloud` + `matplotlib` for word cloud generation.", icon="💡")

    section("🔗 N-Gram Analysis")
    c1, c2 = st.columns(2)
    with c1:
        bi = get_ngrams(df, n=2, top_k=20)
        if not bi.empty:
            st.plotly_chart(chart_ngrams(bi, 2), use_container_width=True, key="content_bigrams")
    with c2:
        tri = get_ngrams(df, n=3, top_k=20)
        if not tri.empty:
            st.plotly_chart(chart_ngrams(tri, 3), use_container_width=True, key="content_trigrams")

    if SKLEARN_AVAILABLE:
        section("🔑 TF-IDF Keywords")
        kw = tfidf_keywords(df, top_n=20)
        if not kw.empty:
            st.plotly_chart(chart_tfidf(kw), use_container_width=True, key="content_tfidf")
            table(kw, "tfidf_keywords.csv")

    section("😄 Emoji Analysis")
    ef = emoji_frequency(df, top_n=25)
    if ef.empty:
        st.info("No emojis found.", icon="ℹ️")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(chart_emojis(ef), use_container_width=True, key="content_emojis")
        with c2:
            table(ef.head(15), "emojis.csv")

    section("📜 Longest Messages")
    table(longest_messages(df), "longest_messages.csv")

    if SKLEARN_AVAILABLE:
        section("📚 Topic Modeling (LDA)")
        topics = topic_modeling(df, n_topics=5)
        if topics:
            cols = st.columns(min(len(topics), 5))
            for i, t in enumerate(topics):
                with cols[i % 5]:
                    st.markdown(f"**Topic {t['topic']}**")
                    for w in t["words"]:
                        st.markdown(f"• {w}")
        else:
            st.info("Not enough data for topic modeling.", icon="ℹ️")


def tab_sentiment(df: pd.DataFrame) -> None:
    if "sentiment_label" not in df.columns:
        st.info("Sentiment analysis not enabled. Toggle it in the sidebar.", icon="ℹ️")
        return

    section("Overall Sentiment")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_sentiment_pie(df), use_container_width=True, key="sent_pie")
    with c2:
        st.plotly_chart(chart_sentiment_timeline(df), use_container_width=True, key="sent_timeline")

    section("📋 Per-Author Sentiment")
    as_df = author_sentiment(df)
    if as_df is not None and not as_df.empty:
        if len(as_df) > 2:
            st.plotly_chart(chart_author_sentiment_radar(as_df), use_container_width=True, key="sent_radar")
        table(as_df, "sentiment_by_author.csv")

    section("🔍 Most Positive & Negative Messages")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Positive Messages**")
        pos = df[df["sentiment_label"] == "Positive"].nlargest(5, "sentiment_compound")
        for _, row in pos.iterrows():
            st.success(f"**{row['author']}**: {row['message'][:200]}")
    with c2:
        st.markdown("**Top Negative Messages**")
        neg = df[df["sentiment_label"] == "Negative"].nsmallest(5, "sentiment_compound")
        for _, row in neg.iterrows():
            st.error(f"**{row['author']}**: {row['message'][:200]}")

    if df["is_toxic"].sum() > 0:
        section("⚠️ Toxic Message Samples")
        toxic = df[df["is_toxic"]][["author", "message"]].head(10)
        toxic.columns = ["Author", "Message"]
        table(toxic, "toxic_messages.csv")


def tab_search(df: pd.DataFrame) -> None:
    section("🔍 Message Search")
    query = st.text_input("Search messages:", placeholder="Type a keyword or phrase…")
    if query:
        results = search_messages(df, query)
        if results.empty:
            st.info(f"No messages found matching **'{query}'**.", icon="🔍")
        else:
            st.success(f"Found **{len(results)}** messages.", icon="✅")
            table(results, f"search_{query[:20]}.csv")

    section("🌍 Language Detection")
    if LANGDETECT_AVAILABLE:
        lang_df = detect_language(df)
        if not lang_df.empty:
            fig = px.bar(lang_df, x="Language", y="Count",
                         color="Count", color_continuous_scale="Blues")
            fig.update_layout(**_BASE, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True, key="search_lang_chart")
            table(lang_df, "languages.csv")
    else:
        st.info("Install `langdetect` for language detection.", icon="💡")

    section("📊 Message Type Breakdown")
    msg_types = {
        "Text Only":    int((~df["is_media"] & ~df["has_url"] & ~df["is_forwarded"]).sum()),
        "Media":        int(df["is_media"].sum()),
        "Links":        int(df["has_url"].sum()),
        "Forwarded":    int(df["is_forwarded"].sum()),
        "Deleted":      int(df["is_deleted"].sum()),
        "Edited":       int(df["is_edited"].sum()),
        "Questions":    int(df["is_question"].sum()),
        "With Emoji":   int((df["emoji_count"] > 0).sum()),
        "With Mention": int(df["has_mention"].sum()),
    }
    type_df = pd.DataFrame(list(msg_types.items()), columns=["Type", "Count"])
    fig = px.bar(type_df, x="Count", y="Type", orientation="h",
                 color="Count", color_continuous_scale="Teal")
    fig.update_layout(**_BASE, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, key="search_msg_type_chart")


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=3)
def load_chat(file_bytes: bytes) -> Optional[pd.DataFrame]:
    """Parse → enrich pipeline, cached by Streamlit."""
    try:
        with st.spinner("📂 Parsing chat file…"):
            raw = parse_chat(file_bytes)
        if raw.empty:
            st.error(
                "⚠️ No messages found.\n\n"
                "Ensure you exported **without media** and the `.txt` is unmodified.",
                icon="🚨",
            )
            return None
        with st.spinner("⚙️ Enriching data…"):
            df = enrich(raw)
        st.success(
            f"✅ Loaded **{len(df):,}** messages from **{df['author'].nunique()}** participants.",
            icon="🎉",
        )
        return df
    except ValueError as exc:
        st.error(f"❌ Encoding error: {exc}", icon="🚨")
        return None
    except Exception as exc:
        st.error(f"❌ Unexpected error: `{exc}`", icon="🚨")
        logger.error(traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Background ── */
.main { background: #07071a; }
.stApp { background: linear-gradient(135deg,#07071a 0%,#0d0d2b 50%,#070720 100%); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#111130 0%,#1a1a40 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s, border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.6);
}
[data-testid="metric-container"] label {
    color: #a5b4fc !important;
    font-size: 0.65rem !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.6rem !important;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 14px;
    background: rgba(99,102,241,0.04);
    padding: 8px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(17,17,48,0.8);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99,102,241,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.82rem;
    color: #64748b;
    padding: 6px 14px;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.4);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d0d2b,#07071a);
    border-right: 1px solid rgba(99,102,241,0.15);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── Buttons ── */
.stDownloadButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity: 0.85 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style='text-align:center;padding:20px 0 10px'>
  <span style='font-size:3rem'>💬</span>
  <h1 style='display:inline;font-size:2.2rem;font-weight:800;
             background:linear-gradient(135deg,#6366f1 0%,#a78bfa 50%,#06b6d4 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin-left:12px;letter-spacing:-0.02em'>
    SmartChat Intelligence Engine
  </h1>
  <p style='color:#64748b;margin-top:8px;font-size:0.9rem;letter-spacing:0.02em'>
    Advanced NLP · Machine Learning · Real-time Analytics
  </p>
</div>
""", unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload WhatsApp Chat Export (.txt)",
        type=["txt"],
        help="Export your chat without media from WhatsApp and upload the .txt file.",
    )

    if not uploaded:
        upload_help()

        # Feature badges
        st.markdown("<br>", unsafe_allow_html=True)
        badges = [
            ("🔬", "NLP Analysis",       "TF-IDF, LDA Topics, N-Grams"),
            ("🤖", "Machine Learning",    "Clustering, Anomaly Detection, Forecasting"),
            ("😊", "Sentiment Analysis",  "VADER per message & author"),
            ("☁️", "Word Cloud",          "Visual frequency maps"),
            ("📊", "50+ Charts",          "Plotly interactive visualizations"),
            ("🔍", "Message Search",      "Full-text search with export"),
        ]
        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(badges):
            with cols[i % 3]:
                st.markdown(
                    f"""<div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px;padding:16px;margin-bottom:10px'>
                    <div style='font-size:1.5rem'>{icon}</div>
                    <div style='font-weight:700;color:#e2e8f0;margin:4px 0'>{title}</div>
                    <div style='color:#64748b;font-size:0.82rem'>{desc}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        return

    # ── Load & enrich ─────────────────────────────────────────────────────────
    df = load_chat(uploaded.read())
    if df is None or df.empty:
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("---")

        authors  = sorted(df["author"].unique().tolist())
        selected = st.selectbox("👤 Filter by Author", ["All"] + authors)

        st.markdown("---")
        st.markdown("**📅 Date Range**")
        min_date = df["datetime"].min().date()
        max_date = df["datetime"].max().date()
        date_range = st.date_input(
            "Select range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        st.markdown("---")
        do_sentiment = False
        if VADER_AVAILABLE:
            do_sentiment = st.toggle(
                "😊 Sentiment Analysis", value=False,
                help="VADER sentiment scoring — may be slow on large datasets.",
            )
        else:
            st.caption("💡 `pip install vaderSentiment` to enable sentiment analysis.")

        st.markdown("---")
        st.markdown(
            f"<small style='color:#64748b'>"
            f"📅 {df['datetime'].min().strftime('%b %d, %Y')} →<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{df['datetime'].max().strftime('%b %d, %Y')}<br>"
            f"👥 {df['author'].nunique()} participants<br>"
            f"💬 {len(df):,} messages<br>"
            f"⚙️ sklearn: {'✅' if SKLEARN_AVAILABLE else '❌'}<br>"
            f"📖 NLTK: {'✅' if NLTK_AVAILABLE else '❌'}<br>"
            f"☁️ WordCloud: {'✅' if WORDCLOUD_AVAILABLE else '❌'}<br>"
            f"🌍 LangDetect: {'✅' if LANGDETECT_AVAILABLE else '❌'}"
            f"</small>",
            unsafe_allow_html=True,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    vdf = df.copy()

    # Date filter
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        vdf = vdf[(vdf["datetime"] >= start) & (vdf["datetime"] <= end + timedelta(days=1))]

    # Author filter
    if selected != "All":
        vdf = vdf[vdf["author"] == selected]

    if vdf.empty:
        st.warning("No messages match the selected filters.", icon="⚠️")
        return

    # ── Optional sentiment enrichment ─────────────────────────────────────────
    if do_sentiment:
        with st.spinner("Running sentiment analysis…"):
            vdf = run_sentiment(vdf)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    ov = overview_stats(vdf)
    render_kpi_strip(ov)
    st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)
    render_kpi_strip_2(ov)
    st.caption(
        f"📅 **{ov.get('first_date','?')}** → **{ov.get('last_date','?')}**  "
        f"&nbsp;·&nbsp; **{ov.get('date_range_days', 0)}** days  "
        f"&nbsp;·&nbsp; **{ov.get('avg_msgs_per_day', 0)}** msgs/day"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_names = ["🏠 Overview", "📈 Activity", "👥 Authors",
                 "💬 Content", "🔍 Search & More"]
    if do_sentiment and "sentiment_label" in vdf.columns:
        tab_names.append("😊 Sentiment")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_overview(vdf, ov)
    with tabs[1]:
        tab_activity(vdf)
    with tabs[2]:
        tab_authors(df if selected == "All" else vdf, selected)
    with tabs[3]:
        tab_content(vdf)
    with tabs[4]:
        tab_search(vdf)
    if len(tabs) > 5:
        with tabs[5]:
            tab_sentiment(vdf)

    # ── Export full data ──────────────────────────────────────────────────────
    with st.expander("💾 Export Processed Dataset"):
        export_cols = [
            "timestamp", "author", "message", "datetime", "date",
            "hour", "day_name", "month", "week", "word_count", "char_count",
            "is_media", "has_url", "has_mention", "emoji_count",
            "is_deleted", "is_edited", "is_forwarded", "is_question",
            "is_toxic", "session_id",
        ]
        if "sentiment_compound" in vdf.columns:
            export_cols += ["sentiment_compound", "sentiment_label"]
        table(
            vdf[[c for c in export_cols if c in vdf.columns]],
            "whatsapp_analysis_pro.csv",
        )

    # ── Library install hints ─────────────────────────────────────────────────
    missing = []
    if not SKLEARN_AVAILABLE:   missing.append("scikit-learn")
    if not WORDCLOUD_AVAILABLE: missing.append("wordcloud matplotlib")
    if not LANGDETECT_AVAILABLE: missing.append("langdetect")
    if not VADER_AVAILABLE:     missing.append("vaderSentiment")
    if missing:
        with st.expander("💡 Unlock more features"):
            st.code(f"pip install {' '.join(missing)}", language="bash")


# ── Run ───────────────────────────────────────────────────────────────────────
main()