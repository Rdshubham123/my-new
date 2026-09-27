"""Unit tests built from real examples shown in the EDA notebook."""
import numpy as np
import pandas as pd
import pytest

from entity_matching.metrics import evaluate
from entity_matching.normalize import normalize_address, normalize_name, skeleton
from entity_matching.decision import Calibrator, expected_f_select, threshold_select
from entity_matching.stage2 import top2_by_group
from entity_matching.utils import run_pool

SAME_ENTITY_NAMES = [
    ("Gold Solutions", "গোল্ড সলিউশনস"),                        # Bengali script
    ("Jain Builders Private Limited", "जैन बिल्डर्स प्राइवेट लिमिटेड"),  # Devanagari
    ("Creative Builders", "ಕ್ರಿಯೇಟಿವ್ ಬಿಲ್ಡರ್ಸ್"),                  # Kannada
    ("Creative Intermediate LLC", "Llc Creative Intermediate"),   # legal moved
    ("Bright Seafood Inc", "Bright Séafood Inc"),                 # accent
    ("South Academy", "5outh Academy"),                           # OCR digit
    ("Washington, Harris and Burrell", "WASHINGT0N, HARRIS AND BURRELL"),
    ("Midwest Applied, LLC", "Midwest Applied, ([LLC])"),         # brackets
    ("Fort Worth Telecommunication", "Fort Worth Telecommunication L.L.C. (ID: 49166)"),
    ("Summit Ministries", "Summit Ministries lnc"),               # l/i OCR in legal form
    ("Bright Automation Enterprises", "Bright Bright Automation Enterprises"),  # repeat
    ("Integrity Services (India) Limited", "*** Integrity Services (India)  Limited"),
]


@pytest.mark.parametrize("a,b", SAME_ENTITY_NAMES)
def test_same_entity_names_share_skeleton(a, b):
    assert normalize_name(a)["skel"] == normalize_name(b)["skel"]


def test_domain_and_alias():
    n = normalize_name("aahanarealty.com")
    assert n["domain"] == 1 and n["core"] == "aahanarealty"
    n = normalize_name("Vantagexylo D.B.A. Global Traders Private Limited")
    assert "global traders" in n["alts"].split("||")
    n = normalize_name("Viovera | www.viovera.com")
    assert n["core"] == "viovera"


def test_legal_forms_canonical():
    assert normalize_name("Shakti Food L.L.P.")["legal"] == "llp"
    assert normalize_name("Viz Exim Private Limited")["legal"] == "ltd pvt"
    assert normalize_name("Thermal & Fils SASU")["legal"] == "sasu"


def test_skeleton_transliteration():
    assert skeleton("marketimg") == skeleton("marketing")
    assert skeleton("0271") == "271"


def test_address_patterns():
    a = normalize_address("##16978 Moore Rd, <NULL>, Andalusia, Alabama", "US")
    b = normalize_address("16978 MOORE ROAD, ANDALUSIA, AL", "US")
    assert a["nums"] == b["nums"] == "16978"
    assert a["state"] == b["state"] == "al"
    assert set(a["alpha"].split()) == set(b["alpha"].split())
    c = normalize_address("PLOT NO B-78/1, AMBERNATH EAST, THANE, महाराष्ट्र", "India")
    d = normalize_address("Plot No. B-78/1, Ambernath East, Thane, MH", "India")
    assert c["nums"] == d["nums"] and c["state"] == d["state"] == "mh"
    e = normalize_address("05131 COPPER MEADOW LN, WEST JORDAN CITY, UT", "US")
    assert e["first_num"] == "5131"
    f = normalize_address("20 bis RUE jules lefebvre, Lille, Hauts-de-France", "France")
    assert f["state"] == "hdf" and "rue" in f["alpha"].split()


def test_threshold_select_one_owner():
    s = np.array([0, 1, 0])
    t = np.array([5, 5, 6])
    p = np.array([0.9, 0.95, 0.2])
    m = threshold_select(s, t, p, thr=0.5, alpha=0.0)
    assert m.tolist() == [False, True, False]  # target 5 goes to its best S1 only


def test_expected_f_select():
    # S1 0: two confident candidates -> take both; S1 1: weak candidate -> empty wins
    s = np.array([0, 0, 0, 1])
    t = np.array([1, 2, 3, 4])
    q = np.array([0.95, 0.9, 0.05, 0.3], np.float32)
    m = expected_f_select(s, t, q, beta=0.5)
    assert m.tolist() == [True, True, False, False]
    # one owner: target 7 claimed by two S1s, only the stronger keeps it
    m = expected_f_select(np.array([0, 1]), np.array([7, 7]), np.array([0.9, 0.8], np.float32))
    assert m.tolist() == [True, False]


def test_top2_and_calibrator():
    top, other = top2_by_group(np.array([0, 0, 1]), np.array([0.2, 0.9, 0.5], np.float32))
    assert top.tolist() == pytest.approx([0.9, 0.9, 0.5])
    assert other.tolist() == pytest.approx([0.9, 0.2, 0.0])
    c = Calibrator().fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    c2 = Calibrator.from_dict(c.to_dict())
    assert c2([0.05, 0.95]).tolist() == pytest.approx([0.0, 1.0])


def test_metric_fbeta():
    truth = pd.DataFrame({"source1_entity_id": ["a", "a"], "matched_id": ["x", "y"]})
    pred = pd.DataFrame({"source1_entity_id": ["a"], "matched_id": ["x"]})
    m = evaluate(pred, truth, ["a", "b"])
    assert m["micro_precision"] == 1.0 and m["micro_recall"] == 0.5
    assert abs(m["micro_fbeta"] - (1.25 * 0.5 / (0.25 + 0.5))) < 1e-9
    assert m["macro_fbeta"] == pytest.approx((m["micro_fbeta"] + 1.0) / 2)  # 'b' empty/empty = 1


def _square(x):
    return x * x


def test_run_pool_parallel_and_serial():
    assert run_pool(_square, range(10), n_jobs=3) == [i * i for i in range(10)]
    assert run_pool(_square, range(3), n_jobs=1) == [0, 1, 4]
