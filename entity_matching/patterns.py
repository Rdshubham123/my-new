"""
Hard-coded normalisation patterns for the Amazon ML Challenge 2026
business entity-resolution task (S1 -> S2/S3 matching).

Every dictionary here comes from a noise pattern observed in the EDA
notebook (dataesplore.ipynb, cells 12-20) or from the countries present
in the test set (US, India, and the unseen France).

All keys are lower-case, ASCII-folded, punctuation-free unless noted.
"""

# ---------------------------------------------------------------------------
# Placeholders that mean "no value" (cell 14: null / <null> / none / nan)
# ---------------------------------------------------------------------------
NULL_TOKENS = {
    "null", "<null>", "none", "nan", "n/a", "na", "nil", "unknown",
    "not available", "not applicable", "-", "--", "0",
}

# ---------------------------------------------------------------------------
# Legal / entity forms -> canonical form.
# Multi-word variants are matched on the space-normalised string first.
# ---------------------------------------------------------------------------
LEGAL_MULTIWORD = [
    # order matters: longest first
    ("limited liability partnership", "llp"),
    ("limited liability company", "llc"),
    ("professional limited liability company", "pllc"),
    ("professional corporation", "pc"),
    ("limited partnership", "lp"),
    ("private limited company", "pvt ltd"),
    ("private limited", "pvt ltd"),
    ("public limited company", "public ltd"),
    ("public limited", "public ltd"),
    ("one person company", "opc"),
    ("societe anonyme", "sa"),
    ("societe par actions simplifiee unipersonnelle", "sasu"),
    ("societe par actions simplifiee", "sas"),
    ("societe a responsabilite limitee", "sarl"),
    ("entreprise unipersonnelle a responsabilite limitee", "eurl"),
    ("societe civile immobiliere", "sci"),
    ("et compagnie", "cie"),
    ("et cie", "cie"),
    ("and company", "co"),
    ("and co", "co"),
    ("pvt ltd", "pvt ltd"),
]

LEGAL_TOKENS = {
    # US
    "llc": "llc", "l.l.c": "llc", "llc.": "llc", "lllc": "llc",
    "inc": "inc", "incorporated": "inc", "incorp": "inc", "inco": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lmtd": "ltd", "limted": "ltd",
    "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "plc": "plc", "lc": "lc", "ltd.": "ltd",
    # India
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd", "prv": "pvt",
    "public": "public", "opc": "opc",
    # France (unseen in train -> hard-coded)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "scp": "scp", "selarl": "selarl",
    "scop": "scop", "gie": "gie", "cie": "cie", "compagnie": "cie",
    "ets": "ets", "etablissements": "ets", "etablissement": "ets",
    "societe": "societe", "ste": "societe",
    # Generic international
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl",
    "spa": "spa", "oy": "oy", "ab": "ab", "pty": "pty",
}

# Native-script legal words (cells 13/15: Indic-script names keep the legal
# form transliterated phonetically).  Replaced BEFORE transliteration.
INDIC_LEGAL = {
    # Devanagari (Hindi / Marathi)
    "प्राइवेट": " private ", "प्रायवेट": " private ", "प्राईवेट": " private ",
    "लिमिटेड": " limited ", "लिमीटेड": " limited ",
    "प्रा.": " pvt ", "प्रा": " pvt ", "लि.": " ltd ", "लि": " ltd ",
    "एलएलपी": " llp ", "कंपनी": " company ", "कम्पनी": " company ",
    "इंक": " inc ", "कॉर्प": " corp ",
    # Bengali
    "প্রাইভেট": " private ", "লিমিটেড": " limited ", "প্রা.": " pvt ",
    "লি.": " ltd ", "এলএলপি": " llp ",
    # Tamil
    "பிரைவேட்": " private ", "லிமிடெட்": " limited ", "எல்எல்பி": " llp ",
    # Telugu
    "ప్రైవేట్": " private ", "లిమిటెడ్": " limited ", "ఎల్ఎల్పి": " llp ",
    # Kannada
    "ಪ್ರೈವೇಟ್": " private ", "ಲಿಮಿಟೆಡ್": " limited ", "ಎಲ್ಎಲ್ಪಿ": " llp ",
    # Gujarati
    "પ્રાઇવેટ": " private ", "લિમિટેડ": " limited ", "પ્રા.": " pvt ",
    "લિ.": " ltd ", "એલએલપી": " llp ",
    # Malayalam
    "പ്രൈവറ്റ്": " private ", "ലിമിറ്റഡ്": " limited ", "എൽഎൽപി": " llp ",
    # Punjabi (Gurmukhi) / Odia – rarer but cheap to include
    "ਪ੍ਰਾਈਵੇਟ": " private ", "ਲਿਮਿਟੇਡ": " limited ",
    "ପ୍ରାଇଭେଟ": " private ", "ଲିମିଟେଡ": " limited ",
}

# Phonetic skeletons of legal words: catches transliterations such as
# "praivet", "piraivet", "limitet", "limitedd" (see normalize.skeleton()).
LEGAL_SKELETON_WORDS = [
    "private", "limited", "pvt", "ltd", "llp", "company", "incorporated",
    "corporation", "praivet", "piraivet", "limitet", "elelpi",
]

# DBA / alias markers (cells 15-16): "X D.B.A. Y", "Korzephdova formerly: Y",
# "Viovera | www.viovera.com".
ALIAS_MARKERS = [
    r"\bd\s*/\s*b\s*/\s*a\b", r"\bd\.?\s?b\.?\s?a\.?\b", r"\bdba\b",
    r"\bt\s*/\s*a\b", r"\btrading\s+as\b", r"\bdoing\s+business\s+as\b",
    r"\bformerly\s+known\s+as\b", r"\bformerly\b", r"\bf\s*/\s*k\s*/\s*a\b",
    r"\bfka\b", r"\ba\s*/\s*k\s*/\s*a\b", r"\baka\b", r"\balso\s+known\s+as\b",
    r"\bsucc(?:essor)?\s+to\b", r"\bnee\b", r"\|",
]

# Junk that the noisy sources prepend/append (cells 13-15).
JUNK_REGEXES = [
    r"\(\s*id\s*[:#]?\s*\d+\s*\)",      # (ID: 49166)
    r"\bid\s*[:#]\s*\d+\b",              # ID: 49166
    r"#\s*\d{3,}\b",                     # #98587
    r"^\W+",                             # ***, --, >>, # at start
    r"\W+$",                             # trailing symbols
    r"\bm\s*/\s*s\b\.?",                  # M/s  (Indian "Messrs"), anywhere
    r"^\s*messrs\.?\s+",
    r"\bwww\.",                          # www.
    r"https?://",
]

# Words that carry no identity in a business name.
NAME_STOPWORDS = {
    "the", "of", "and", "a", "an", "for", "at", "in", "on", "by",
    # French
    "de", "la", "le", "les", "du", "des", "et", "l", "d", "au", "aux",
    # Indian honorifics that noisy sources add/drop ("Sri", "Shri", "M/s")
    "m/s", "ms", "messrs",
}
HONORIFICS = {"sri", "shri", "shree", "smt", "dr", "mr", "mrs", "st"}

# Top-level domains stripped from names written as URLs ("aahanarealty.com").
TLDS = {
    "com", "net", "org", "in", "co", "io", "biz", "info", "us", "fr",
    "co.in", "org.in", "net.in", "edu", "gov", "c0m", "corn",
}

# ---------------------------------------------------------------------------
# OCR / "leet" confusions inside alphabetic tokens (cell 15/16: 5outh,
# WASHINGT0N, lnc, FAMlLY, silvergrill.c0m, Dermato1ogy, MEMORlAL).
# Applied symmetrically to BOTH sides to build an OCR-folded comparison key.
# ---------------------------------------------------------------------------
OCR_DIGIT_TO_CHAR = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t",
    "8": "b", "9": "g", "@": "a", "$": "s", "|": "l", "!": "l",
}
# After digit fixing, collapse visually confusable letters.
OCR_CHAR_FOLD = [("rn", "m"), ("vv", "w"), ("cl", "d"), ("l", "i"), ("j", "i")]

# ---------------------------------------------------------------------------
# Address: US states (name <-> USPS code)
# ---------------------------------------------------------------------------
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr", "guam": "gu", "virgin islands": "vi",
}

# ---------------------------------------------------------------------------
# Address: Indian states / UTs (name, common code, native-script spelling)
# ---------------------------------------------------------------------------
INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
}
INDIA_STATE_CODE_ALIASES = {"ts": "tg", "or": "od", "ua": "uk", "ct": "cg"}

# Native-script state names seen in S2/S3 addresses (cell 13).
INDIA_STATES_NATIVE = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "राजस्थान": "rj", "मध्य प्रदेश": "mp",
    "उत्तर प्रदेश": "up", "बिहार": "br", "हरियाणा": "hr", "पंजाब": "pb",
    "उत्तराखंड": "uk", "झारखंड": "jh", "छत्तीसगढ़": "cg", "हिमाचल प्रदेश": "hp",
    "गोवा": "ga", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn", "తెలంగాణ": "tg",
    "ఆంధ్ర ప్రదేశ్": "ap", "ఆంధ్రప్రదేశ్": "ap", "পশ্চিমবঙ্গ": "wb",
    "ગુજરાત": "gj", "കേരളം": "kl", "ଓଡ଼ିଶା": "od", "ਪੰਜਾਬ": "pb",
    "অসম": "as", "आसाम": "as",
}

# ---------------------------------------------------------------------------
# France regions / departments present in test (cell 12, descriptive only)
# ---------------------------------------------------------------------------
FRANCE_REGIONS = {
    "hauts de france": "hdf", "nouvelle aquitaine": "naq",
    "pays de la loire": "pdl", "ile de france": "idf", "bretagne": "bre",
    "normandie": "nor", "grand est": "ges", "occitanie": "occ",
    "auvergne rhone alpes": "ara", "provence alpes cote d azur": "pac",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl",
    "corse": "cor", "nord": "d59", "pas de calais": "d62", "gironde": "d33",
    "loire atlantique": "d44",
}

# ---------------------------------------------------------------------------
# City aliases (old/new names, common misspellings)
# ---------------------------------------------------------------------------
CITY_ALIASES = {
    "bengaluru": "bangalore", "bangaluru": "bangalore", "bombay": "mumbai",
    "madras": "chennai", "calcutta": "kolkata", "gurugram": "gurgaon",
    "thiruvananthapuram": "trivandrum", "poona": "pune", "baroda": "vadodara",
    "prayagraj": "allahabad", "vizag": "visakhapatnam",
    "vishakhapatnam": "visakhapatnam", "vishakapatnam": "visakhapatnam",
    "mysuru": "mysore", "mangaluru": "mangalore", "cochin": "kochi",
    "belagavi": "belgaum", "kalaburagi": "gulbarga", "cawnpore": "kanpur",
    "benares": "varanasi", "banaras": "varanasi", "simla": "shimla",
    "ahmedbad": "ahmedabad", "amdavad": "ahmedabad", "secunderabad": "hyderabad",
    "hubballi": "hubli", "tiruchirappalli": "trichy", "tiruchirapalli": "trichy",
    "puducherry": "pondicherry", "nw delhi": "new delhi", "navi mumbai": "navi mumbai",
    "gautam budh nagar": "gautam buddha nagar", "goutam budd nagar": "gautam buddha nagar",
    "noida": "gautam buddha nagar",
}

# ---------------------------------------------------------------------------
# Street / thoroughfare types (USPS Pub 28 + Indian + French) -> canonical
# ---------------------------------------------------------------------------
STREET_TYPES = {
    # US
    "street": "st", "str": "st", "st": "st", "saint": "st",  # "25 PALMER SAINT"
    "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave",
    "aven": "ave", "avn": "ave", "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "crt": "ct",
    "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "bd": "blvd",
    "place": "pl", "pl": "pl", "terrace": "ter", "ter": "ter", "terr": "ter",
    "trail": "trl", "trl": "trl", "circle": "cir", "cir": "cir", "circ": "cir",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy", "highway": "hwy",
    "hwy": "hwy", "square": "sq", "sq": "sq", "way": "way", "wy": "way",
    "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy",
    "turnpike": "tpke", "tpke": "tpke", "crossing": "xing", "xing": "xing",
    "point": "pt", "pt": "pt", "pointe": "pt", "heights": "hts", "hts": "hts",
    "mount": "mt", "mt": "mt", "mountain": "mtn", "mtn": "mtn",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "plaza": "plz", "plz": "plz",
    "alley": "aly", "aly": "aly", "loop": "loop", "run": "run", "row": "row",
    "pike": "pike", "path": "path", "walk": "walk", "cove": "cv", "cv": "cv",
    "creek": "crk", "crk": "crk", "ridge": "rdg", "rdg": "rdg",
    "hollow": "holw", "holw": "holw", "valley": "vly", "vly": "vly",
    "junction": "jct", "jct": "jct", "estates": "est", "est": "est",
    "station": "sta", "sta": "sta", "township": "twp", "twp": "twp",
    "townhsip": "twp",
    # India
    "marg": "marg", "mg": "marg", "nagar": "nagar", "ngr": "nagar",
    "colony": "colony", "col": "colony", "sector": "sector", "sec": "sector",
    "phase": "phase", "ph": "phase", "block": "block", "blk": "block",
    "layout": "layout", "extension": "extn", "extn": "extn", "ext": "extn",
    "chowk": "chowk", "gali": "gali", "mohalla": "mohalla", "bazar": "bazaar",
    "bazaar": "bazaar", "cross": "cross", "main": "main", "salai": "salai",
    "industrial": "indl", "indl": "indl", "ind": "indl", "estate": "est",
    "complex": "cplx", "cplx": "cplx", "building": "bldg", "bldg": "bldg",
    "bldng": "bldg", "tower": "twr", "twr": "twr", "apartment": "apt",
    "apartments": "apt", "apts": "apt", "apt": "apt", "society": "soc",
    "soc": "soc", "chs": "soc", "near": "nr", "nr": "nr", "opposite": "opp",
    "opp": "opp", "behind": "bhd", "bh": "bhd", "post": "po", "po": "po",
    "village": "vill", "vill": "vill", "vil": "vill", "taluka": "tal",
    "tal": "tal", "tehsil": "tal", "district": "dist", "dist": "dist",
    "distt": "dist", "railway": "rly", "rly": "rly",
    # France
    "rue": "rue", "r": "rue", "allee": "all", "all": "all", "impasse": "imp",
    "imp": "imp", "chemin": "che", "che": "che", "ch": "che", "route": "rte",
    "rte": "rte", "quai": "quai", "cours": "crs", "crs": "crs", "residence": "res",
    "res": "res", "lieu dit": "ld", "lieudit": "ld", "ld": "ld", "faubourg": "fbg",
    "fbg": "fbg", "passage": "pass", "pass": "pass", "sentier": "sen",
    "hameau": "ham", "ham": "ham", "zone": "zone", "za": "za", "zi": "zi",
    "zac": "zac", "bis": "bis", "ter.": "ter", "sainte": "ste", "ste": "ste",
    "cedex": "",
}

DIRECTIONALS = {
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw", "n": "n",
    "s": "s", "e": "e", "w": "w", "ne": "ne", "nw": "nw", "se": "se",
    "sw": "sw", "nord": "n", "sud": "s", "est": "e", "ouest": "w",
}

# Unit / house labels that precede a number (cell 14): the label itself is
# dropped, the number is kept.
UNIT_LABELS = {
    "unit", "apt", "apartment", "suite", "ste", "room", "rm", "floor", "fl",
    "flr", "office", "off", "shop", "plot", "flat", "house", "door", "no",
    "number", "num", "nos", "hno", "h.no", "hn", "dno", "d.no", "fno", "f.no",
    "khasra", "kh", "survey", "sr", "sy", "s.no", "sno", "ward", "wing",
    "bldg", "building", "block", "pmb", "box", "po box", "pobox", "#",
    "premises", "cabin", "gala", "stall", "site", "khata", "cts", "rs",
    "gat", "gut", "pl", "bat", "batiment", "etage", "appt", "porte",
}

ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12", "ground": "0", "gf": "0",
    "premier": "1", "deuxieme": "2", "troisieme": "3", "rdc": "0",
}

ADDRESS_STOPWORDS = {
    "of", "the", "and", "at", "in", "near", "city", "town", "de", "la", "le",
    "du", "des", "les", "l", "d", "s/o", "c/o", "w/o", "d/o", "county",
    "region", "india", "usa", "us", "france", "united", "states",
}
