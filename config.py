# ============================================================
#  EDIT THIS FILE WITH YOUR INFO BEFORE RUNNING
# ============================================================

USER_INFO = {
    "name":  "Eric Paulson",
    "email": "epaulson@acesinc1.com",
    "phone": "347-633-2712",

    "message": (
        "Hello,\n"
        "My name is Eric Paulson, and I work in the marketing department at a sports agency called ACES Inc. "
        "We represent over 50 current Major League Baseball players and have been in the business over 35 years.\n"
        "https://acesincbaseball.com\n"
        "Victor Caratini, catcher for the Minnesota Twins, has expressed interest in securing a vehicle for the "
        "upcoming season, and we wanted to see if your dealership might be open to a partnership opportunity. "
        "Victor is a nine-year MLB veteran who just signed a two-year contract with the Twins this past January, "
        "bringing a wealth of experience and a strong community presence to the Minneapolis area.\n"
        "A switch-hitting catcher and a native of Puerto Rico, Victor is bilingual and carries broad appeal across "
        "the Twins fanbase. In similar collaborations, our clients have provided autographed memorabilia, personal "
        "appearances, social media or advertising features, and game tickets in exchange for vehicle use.\n"
        "We're also happy to explore creative or custom arrangements that align with your dealership's marketing "
        "goals, whether that's a meet and greet event, signage, or co-branded content. If this sounds like something "
        "your team would consider, I'd love to discuss the details further at your convenience. Thank you for your "
        "time and I look forward to the possibility of working together."
    ),

    "zip": "11201",
}

# ---------------------------------------------------------------
# OPTIONAL: Add specific dealership URLs you already know about.
# Leave the list empty if you just want auto-discovery.
# ---------------------------------------------------------------
EXTRA_DEALERSHIPS = [
]

# ---------------------------------------------------------------
# SEARCH QUERIES
# Uncomment the makes you want to target for a given run.
# MAX_PER_QUERY = how many dealerships to pull per query.
# ---------------------------------------------------------------
SEARCH_QUERIES = [
    "land rover dealership minneapolis",
    "jeep dealership minneapolis",
    "Ford dealership Minneapolis",
    "Chevy dealership Minneapolis",
    "mercedes dealership minneapolis",
    "bmw dealership minneapolis",
    "audi dealership minneapolis",
    "lincoln dealership minneapolis",
    "Toyota dealership Minneapolis",
]

MAX_PER_QUERY = 10
DELAY_BETWEEN_SUBMISSIONS = 3

# ---------------------------------------------------------------
# OLLAMA MODEL
# qwen3:14b — native tool calling, strong instruction following, fits 4080 Super
# Fallback: qwen2.5:14b or llama3.1:8b if you want something faster
# ---------------------------------------------------------------
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_BASE_URL = "http://localhost:11434"
USE_VISION = False
