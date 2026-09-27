"""
Synthetic data in the exact challenge layout, reproducing the noise found
in the EDA notebook.  Used for smoke-testing the pipeline end-to-end (the
real Kaggle data is not bundled).  Scores on synthetic data are NOT an
estimate of leaderboard scores.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

US_W1 = ["Summit", "Harbor", "Meridian", "Pioneer", "Cascade", "Sterling", "Keystone", "Apex",
         "Coastal", "Highland", "Silver", "Golden", "Bright", "Heritage", "Metro", "Blue",
         "Liberty", "Frontier", "Evergreen", "Oakwood", "Riverside", "Granite", "Beacon", "Vista"]
US_W2 = ["Pediatric", "Dental", "Eye", "Chiropractic", "Family", "Urology", "Vision",
         "Orthopedic", "Seafood", "Auto Body", "Printing", "Realty", "Capital", "Logistics",
         "Consulting", "Cargo", "Marketing", "Software", "Roofing", "Insurance"]
US_W3 = ["Group", "Associates", "Partners", "Center", "Care", "Clinic", "Services", "Holdings",
         "Solutions", "Works"]
US_LEGAL = ["LLC", "Inc", "Corp", "Co", "LP", "PC", "PLLC", "Ltd", ""]
SURNAMES = ["Smith", "Jones", "Garcia", "Kern", "Delavega", "Hawkins", "Brunson", "Murphy",
            "Townsend", "Graham", "Figueroa", "Prater", "Younker", "Wellman", "Holloway"]
US_STREETS = ["Westchester", "Ellis", "Montebello", "Cameron", "Elm", "Stardust", "Pierpont",
              "Mulberry", "Forest", "Moore", "Greyrock", "Meadow", "Allison", "Bell", "Oak",
              "Maple", "Cedar", "Lincoln", "Washington", "Park"]
US_TYPES = [("Street", "St"), ("Road", "Rd"), ("Drive", "Dr"), ("Avenue", "Ave"),
            ("Lane", "Ln"), ("Court", "Ct"), ("Trail", "Trl"), ("Boulevard", "Blvd")]
US_CITIES = [("High Point", "NC", "North Carolina"), ("Tahlequah", "OK", "Oklahoma"),
             ("Phoenix", "AZ", "Arizona"), ("Dundalk", "MD", "Maryland"),
             ("Cleveland", "OH", "Ohio"), ("Houston", "TX", "Texas"),
             ("Brooklyn", "NY", "New York"), ("Springfield", "IL", "Illinois"),
             ("Raleigh", "NC", "North Carolina"), ("Eugene", "OR", "Oregon")]

IN_W1 = ["Shakti", "Lakshmi", "Supreme", "Sky", "Green", "Arihant", "Jain", "Global", "Royal",
         "Modern", "Sun", "Krishna", "Balaji", "Great", "Prime", "Apex", "Shivam", "Om"]
IN_W2 = ["Technology", "Builders", "Consultants", "Foods", "Infra", "Marketing", "Ventures",
         "Enterprises", "Exports", "Solutions", "Estate", "Agro", "Finance"]
IN_LEGAL = ["Private Limited", "Pvt Ltd", "Limited", "LLP", "Pvt. Ltd.", ""]
DEVANAGARI = {
    "Shakti": "शक्ति", "Lakshmi": "लक्ष्मी", "Supreme": "सुप्रीम", "Sky": "स्काई",
    "Green": "ग्रीन", "Arihant": "अरिहंत", "Jain": "जैन", "Global": "ग्लोबल", "Royal": "रॉयल",
    "Modern": "मॉडर्न", "Sun": "सन", "Krishna": "कृष्णा", "Balaji": "बालाजी", "Great": "ग्रेट",
    "Prime": "प्राइम", "Apex": "एपेक्स", "Shivam": "शिवम", "Om": "ओम",
    "Technology": "टेक्नोलॉजी", "Builders": "बिल्डर्स", "Consultants": "कंसल्टेंट्स",
    "Foods": "फूड्स", "Infra": "इंफ्रा", "Marketing": "मार्केटिंग", "Ventures": "वेंचर्स",
    "Enterprises": "एंटरप्राइजेज", "Exports": "एक्सपोर्ट्स", "Solutions": "सॉल्यूशंस",
    "Estate": "एस्टेट", "Agro": "एग्रो", "Finance": "फाइनेंस",
    "Private": "प्राइवेट", "Limited": "लिमिटेड", "Pvt": "प्रा.", "Ltd": "लि.", "LLP": "एलएलपी",
}
IN_AREAS = ["Gulmohar Colony", "Karve Nagar", "Anand Nagar", "Jayanagar", "Kamala Nagar",
            "Andheri East", "Sector 42", "Rajaji Nagar", "Salt Lake", "Banjara Hills"]
IN_CITIES = [("Mumbai", "MH", "Maharashtra", "महाराष्ट्र"), ("Pune", "MH", "Maharashtra", "महाराष्ट्र"),
             ("Bangalore", "KA", "Karnataka", "ಕರ್ನಾಟಕ"), ("Chennai", "TN", "Tamil Nadu", "தமிழ்நாடு"),
             ("Hyderabad", "TG", "Telangana", "తెలంగాణ"), ("Kolkata", "WB", "West Bengal", "পশ্চিমবঙ্গ"),
             ("Ahmedabad", "GJ", "Gujarat", "ગુજરાત"), ("New Delhi", "DL", "Delhi", "दिल्ली")]
IN_LABELS = ["No.", "H.No", "Plot No", "Door No", "Flat No", "Shop No", "Office No"]

FR_W1 = ["Amicale", "Club", "Ecole", "Maison", "Comite", "Union", "Groupe", "Centre",
         "Pharmacie", "Institut", "Association", "Foyer"]
FR_W2 = ["Sportive", "Culturelle", "des Parents", "Bouliste", "du Lac", "Saint-Michel",
         "Laique", "Rotary", "des Amis", "Primaire"]
FR_LEGAL = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", ""]
FR_STREETS = ["Pierre Dignac", "Parmentier", "Lachassaigne", "de Cassel", "du Chaufour",
              "Jules Lefebvre", "Roger Salengro", "du President Wilson", "de la Renaudiere"]
FR_TYPES = [("Rue", "R"), ("Avenue", "Av"), ("Boulevard", "BD"), ("Allee", "All"),
            ("Impasse", "Imp"), ("Chemin", "Ch")]
FR_CITIES = [("Bordeaux", "Nouvelle-Aquitaine"), ("Nantes", "Pays de la Loire"),
             ("Lille", "Hauts-de-France"), ("Dunkerque", "Hauts-de-France"),
             ("Calais", "Hauts-de-France"), ("Pessac", "Nouvelle-Aquitaine")]

MATCH_COUNT_DIST = [(0, 5.6), (1, 5.4), (2, 17.0), (3, 24.1), (4, 21.9), (5, 14.6),
                    (6, 7.5), (7, 2.9), (8, 0.85), (9, 0.2), (10, 0.03)]
ACCENTS = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "c": "ç"}
OCR = {"o": "0", "l": "1", "s": "5", "i": "l", "e": "3"}


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.used_ids = set()

    def uid(self, prefix):
        while True:
            i = self.r.randint(10_000, 999_999_999)
            if i not in self.used_ids:
                self.used_ids.add(i)
                return f"{prefix}-{i}"

    # ---------------- clean entities ----------------
    def entity(self, country):
        r = self.r
        if country == "US":
            if r.random() < 0.35:  # generic -> hard negatives (cell 20)
                words = [r.choice(US_W2[:8]), r.choice(["Group", "Associates", "Center", "Care"])]
            elif r.random() < 0.2:
                a, b = r.sample(SURNAMES, 2)
                words = [f"{a},", b, "&", r.choice(SURNAMES)]
            else:
                words = [r.choice(US_W1), r.choice(US_W2)] + ([r.choice(US_W3)] if r.random() < .6 else [])
            legal = r.choice(US_LEGAL)
            city, code, state = r.choice(US_CITIES)
            num = str(r.randint(1, 29999))
            st, _ = r.choice(US_TYPES)
            addr = {"num": num, "street": r.choice(US_STREETS), "type": st, "city": city,
                    "code": code, "state": state,
                    "unit": (f"Unit {r.randint(1, 900)}" if r.random() < 0.2 else "")}
        elif country == "India":
            words = [r.choice(IN_W1), r.choice(IN_W2)]
            if r.random() < 0.3:
                words.insert(1, r.choice(IN_W1))
            legal = r.choice(IN_LEGAL)
            city, code, state, native = r.choice(IN_CITIES)
            n = str(r.randint(1, 999))
            num = n + (f"/{r.randint(1, 99)}" if r.random() < 0.5 else "")
            addr = {"label": r.choice(IN_LABELS), "num": num, "area": r.choice(IN_AREAS),
                    "city": city, "code": code, "state": state, "native": native,
                    "floor": (r.choice(["1st Floor", "2nd Floor", "Ground Floor"]) if r.random() < .3 else "")}
        else:
            words = [r.choice(FR_W1), r.choice(FR_W2)]
            legal = r.choice(FR_LEGAL)
            city, region = r.choice(FR_CITIES)
            st, _ = r.choice(FR_TYPES)
            addr = {"num": str(r.randint(1, 400)), "bis": r.random() < 0.1, "type": st,
                    "street": r.choice(FR_STREETS), "city": city, "region": region}
        return {"country": country, "words": words, "legal": legal, "addr": addr}

    def render_name(self, e):
        return " ".join(e["words"] + ([e["legal"]] if e["legal"] else [])).replace(" ,", ",")

    def render_addr(self, e, noisy=False):
        r, a, c = self.r, e["addr"], e["country"]
        if c == "US":
            typ = a["type"]
            if noisy and r.random() < 0.5:
                typ = dict(US_TYPES)[typ]
            num = a["num"]
            if noisy and r.random() < 0.1:
                num = "0" + num
            if noisy and r.random() < 0.05:
                num = "#" + num
            state = a["code"] if (not noisy or r.random() < 0.5) else a["state"]
            comps = [f"{num} {a['street']} {typ}"] + ([a["unit"]] if a["unit"] else []) + [a["city"], state]
        elif c == "India":
            lab = a["label"] if not noisy else r.choice(IN_LABELS + [a["label"]] * 2)
            state = a["state"]
            if noisy:
                state = r.choice([a["state"], a["code"], a["native"]])
            comps = [f"{lab} {a['num']}"] + ([a["floor"]] if a["floor"] else []) + [a["area"], a["city"], state]
        else:
            typ = a["type"]
            if noisy and r.random() < 0.4:
                typ = dict(FR_TYPES)[typ]
            num = a["num"] + (" bis" if a["bis"] else "")
            comps = [f"{num} {typ} {a['street']}", a["city"], a["region"]]
        if noisy:
            if r.random() < 0.35:
                r.shuffle(comps)
            if r.random() < 0.08:
                comps.insert(r.randint(0, len(comps)), r.choice(["NULL", "null", "<NULL>"]))
            if r.random() < 0.1 and len(comps) > 2:
                comps.pop(r.randrange(len(comps)))
            s = ", ".join(comps)
            if r.random() < 0.3:
                s = s.upper()
            if r.random() < 0.15:
                s = self.typo(s)
            if r.random() < 0.035:
                s = ""
            return s
        return ", ".join(comps)

    # ---------------- noise ops ----------------
    def typo(self, s):
        r = self.r
        if len(s) < 4:
            return s
        i = r.randrange(1, len(s) - 1)
        op = r.random()
        if op < 0.3:
            return s[:i] + s[i + 1:]
        if op < 0.6:
            return s[:i] + r.choice("abcdefghilmnorstu") + s[i:]
        if op < 0.8:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s[:i] + r.choice("aeiou") + s[i + 1:]

    def noisy_name(self, e):
        r = self.r
        words = list(e["words"])
        legal = e["legal"]
        c = e["country"]
        if r.random() < 0.02:  # name replaced entirely (cell 17: "Kelopyrahalo")
            return r.choice(["Kelopyrahalo", "Noviveo", "Ectoecto", "Viovera", "Jaxzeta"])
        if c == "India" and r.random() < 0.2 and all(w in DEVANAGARI for w in words):
            parts = [DEVANAGARI[w] for w in words]
            for lw in legal.replace(".", "").split():
                parts.append(DEVANAGARI.get(lw, lw))
            if r.random() < 0.3:  # mixed script
                parts[0] = words[0]
            return " ".join(parts)
        ops = r.sample(range(14), k=r.choice([1, 1, 2, 2, 3]))
        for op in ops:
            if op == 0 and legal:
                legal = r.choice({"US": US_LEGAL, "India": IN_LEGAL, "France": FR_LEGAL}[c])
            elif op == 1 and legal:
                legal = r.choice([f"({legal})", f"[{legal}]", f"([{legal}])", legal.upper()])
            elif op == 2 and legal:
                words = [legal] + words
                legal = ""
            elif op == 3:
                i = r.randrange(len(words))
                words[i] = self.typo(words[i])
            elif op == 4:
                i = r.randrange(len(words))
                words[i] = "".join(OCR.get(ch, ch) if r.random() < 0.3 else ch for ch in words[i])
            elif op == 5:
                i = r.randrange(len(words))
                words[i] = "".join(ACCENTS.get(ch, ch) if r.random() < 0.2 else ch for ch in words[i])
            elif op == 6:
                i = r.randrange(len(words))
                words.insert(i, words[i])
            elif op == 7 and len(words) > 1:
                r.shuffle(words)
            elif op == 8:
                name = "".join(w.strip(",&") for w in words).lower()
                return name + r.choice([".com", ".Com", ".c0m", ".net"])
            elif op == 9:
                alias = r.choice(["Vantagexylo D.B.A.", "Korzephdova formerly:", "T/A"])
                return f"{alias} {' '.join(words)} {legal}".strip()
            elif op == 10:
                words.append(r.choice(["Partners", "Center", "Group", "Co", "Enterprises"]))
            elif op == 11:
                return f"{' '.join(words)} {legal} (ID: {r.randint(10000, 99999)})".strip()
            elif op == 12:
                words = [r.choice(["***", "--", ">>", "M/s"])] + words
            elif op == 13:
                legal = ""
        s = " ".join(words + ([legal] if legal else []))
        rr = r.random()
        if rr < 0.2:
            s = s.upper()
        elif rr < 0.3:
            s = s.lower()
        elif rr < 0.35:
            s = s.replace(" ", "  ", 1)
        return s

    # ---------------- dataset ----------------
    def build(self, n_s1, countries):
        s1, s2, s3, gt = [], [], [], []
        counts, weights = zip(*MATCH_COUNT_DIST)
        for _ in range(n_s1):
            c = self.r.choices(countries[0], weights=countries[1])[0]
            e = self.entity(c)
            sid = self.uid("S1")
            s1.append((sid, self.render_name(e), self.render_addr(e), c))
            k = self.r.choices(counts, weights=weights)[0]
            ids = []
            for _ in range(k):
                src = self.r.choice(["S2", "S3"])
                tid = self.uid(src)
                row = (tid, self.noisy_name(e), self.render_addr(e, noisy=True), c)
                (s2 if src == "S2" else s3).append(row)
                ids.append(tid)
            gt.append((sid, ",".join(ids)))
        # distractor targets: entities absent from S1 (~30 % of targets)
        n_dis = int(0.3 * (len(s2) + len(s3)))
        for _ in range(n_dis):
            c = self.r.choices(countries[0], weights=countries[1])[0]
            e = self.entity(c)
            src = self.r.choice(["S2", "S3"])
            row = (self.uid(src), self.noisy_name(e), self.render_addr(e, noisy=True), c)
            (s2 if src == "S2" else s3).append(row)
        cols = ["entity_id", "business_name", "business_address", "country"]
        self.r.shuffle(s2)
        self.r.shuffle(s3)
        return (pd.DataFrame(s1, columns=cols), pd.DataFrame(s2, columns=cols),
                pd.DataFrame(s3, columns=cols),
                pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]))


def write_synthetic(out_dir, n_train=6000, n_test=3000, seed=0):
    out = Path(out_dir)
    g = Gen(seed)
    for split, n, countries in [("train", n_train, (["US", "India"], [60, 40])),
                                ("test", n_test, (["US", "India", "France"], [38, 47, 15]))]:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        s1, s2, s3, gt = g.build(n, countries)
        for name, df in [("source1", s1), ("source2", s2), ("source3", s3), ("ground_truth", gt)]:
            df.to_csv(d / f"{split}_{name}.tsv", sep="\t", index=False)
        print(f"{split}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} links={gt.matched_entity_ids.str.count('S').sum():,}")
