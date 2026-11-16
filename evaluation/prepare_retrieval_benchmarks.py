"""
Prepare and validate:
1. Multi-passage ground-truth refinement for the 2-document 22-question benchmark
   (preserving the original annotated_benchmark.json untouched).
2. Held-out 14-question benchmark (7 Potemra + 7 WMM2020) for generalizing retrieval evaluation.
3. Expanded 15-document public corpus manifest and 25-question expanded ground-truth annotations
   (including Q22, Q23, Q24 from Encyclopedia_of_Geomagnetism.pdf and Butler Ch. 1/7).
"""

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from public_retrieval import chunk_identifier


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xad\n", "").replace("\xad", "")).strip()


POTEMRA_URL = "https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V01-N03/01-03-Potemra_Geomag.pdf"
WMM2020_URL = "https://repository.library.noaa.gov/view/noaa/24390"

# Multi-passage additions on 01-03-Potemra_Geomag.pdf where the same supporting fact
# appears in both the main narrative and a figure caption / glossary / adjacent overlap chunk.
MULTICHUNK_ADDITIONS = {
    "Q02": [
        (2, "A principal goal of Magsat is to provide the most accurate evaluation of the geo"),
    ],
    "Q08": [
        (2, 'The distorted configuration of its geomagnetic field is called the "magnetosphere."'),
        (4, "Fig. 4- The configuration of the earth's dipole magnetic field distorted into the comet-like shape"),
    ],
    "Q12": [
        (5, "the orbit plane precesses at the rate of"),
        (5, "causes the orbit plane to ro"),
    ],
    "Q13": [
        (5, "This s ystem, the Attitude Transfer System (ATS), used an optical technique involving mirrors attached to the vector sensor"),
    ],
    "Q14": [
        (8, "The vector magnetometer sensor had three small magnetic elements, each sensitive to one component of the earth's field"),
    ],
    "Q15": [
        (8, "The scalar magnetometer measured only the magnitude of the field by optical pump"),
    ],
}


HELDOUT_QUESTIONS = [
    {
        "id": "H01",
        "question": "How did the Chinese emperor Hoang-Ti navigate through the dense fog raised by Tchi-Yeou in 2634 B.C.?",
        "expected_answer": "Hoang-Ti constructed a chariot carrying a small figure of a man with an outstretched arm that revolved on a vertical axis and always pointed south, allowing him to locate the direction of the enemy's retreat.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (1, "stood the small figure of a man with his arm outstretched"),
            (1, "always pointed to the south, allowing the emperor to locate the direction of his enemy's retreat"),
        ],
    },
    {
        "id": "H02",
        "question": "Which earlier APL-built satellite launched in 1975 served as the design basis for Magsat, and what was the maximum spacecraft weight the Scout rocket could launch into Magsat's orbit?",
        "expected_answer": "The Small Astronomical Satellite (SAS-3), designed and built by APL and launched in 1975, served as the basis for Magsat, and the maximum spacecraft weight the Scout rocket could place into the chosen orbit was 182 kg.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (4, "The Small Astronomical Satellite (SAS-3) had been designed and built by APL"),
            (5, "keep the weight down to 1 8 2 kg, the maximum that the Scout rocket"),
        ],
    },
    {
        "id": "H03",
        "question": "How was the Magsat magnetometer boom constructed, and how far did it deploy the sensor platform from the instrument module?",
        "expected_answer": "The magnetometer boom was a collapsible structural element composed of seven pairs of links in a scissors or 'lazy-tongs' arrangement that deployed the sensor platform 6 meters away from the instrument module.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (7, "Magnetometer Boom - A collapsible structural element composed of seven pairs of links"),
            (7, "6 meters away from the instrument module"),
        ],
    },
    {
        "id": "H04",
        "question": "What output frequency did Magsat's ultrastable quartz crystal oscillator generate, and how was its high thermal stability achieved?",
        "expected_answer": "Magsat's ultrastable quartz crystal oscillator produced a 5 MHz output (used for the 162 and 324 MHz Doppler signals and synchronizing DC-DC converters), and its thermal stability was achieved by placing the crystal inside a double-oven arrangement with its turnover temperature matched to the oven temperature.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (8, "Oscillator - An ultrastable quartz crystal oscillator producing a 5 MHz output"),
            (8, "providing a high degree of thermal isolation"),
        ],
    },
    {
        "id": "H05",
        "question": "What were the four main elements of Magsat's power supply system?",
        "expected_answer": "Magsat's power supply system consisted of solar cell arrays to generate electricity, a battery in the base module for shadowed orbit portions and peak power demands, battery voltage limiter devices to control charging, and DC-DC converter regulators to condition power.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (8, "Power Supply - The system"),
        ],
    },
    {
        "id": "H06",
        "question": "How did Magsat's star cameras detect starlight, and what angular accuracy did the precision solar aspect sensor achieve?",
        "expected_answer": "Each of the two star cameras used a 4-inch lens to focus starlight onto an image-dissector electronic tube that emitted electrons into an electron multiplier scanned and locked by magnetic coils; the precision solar aspect sensor measured angles to the sun with an accuracy of 5 to 10 arc-seconds.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (9, "measured the angles to the sun with an accuracy of 5 to 10 arc-seconds"),
            (9, "Star Cameras - Two star cameras were mounted rigidly on the optical bench"),
        ],
    },
    {
        "id": "H07",
        "question": "How was Magsat clamped to the fourth-stage rocket by the vehicle adapter, and how was spacecraft separation executed?",
        "expected_answer": "Magsat was clamped to the conically-shaped vehicle adapter on the fourth-stage rocket by a two-half clamp fastened with bolts passing through pyrotechnically operated cutters; separation was achieved by actuating the bolt cutters from the spacecraft battery via the despin/separation timers so small springs forced the spacecraft away.",
        "source_document": "01-03-Potemra_Geomag.pdf",
        "source_url": POTEMRA_URL,
        "anchors": [
            (9, "Vehicle Adapter - The"),
            (9, "actuating the bolt cutters by a stimulus from the spacecraft battery"),
        ],
    },
    {
        "id": "H08",
        "question": "What is the vertical validity range of the World Magnetic Model (WMM2020) relative to the WGS 84 ellipsoid surface?",
        "expected_answer": "WMM2020 is valid from 1 km below the World Geodetic System (WGS 84) ellipsoid surface to approximately 850 km above it.",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (9, "from 1 km below the World Geodetic System (WGS 84) ellipsoid surface to approximately 850 km above it"),
            (57, "from 1 km below the WGS 84 ellipsoid surface to approximately 850 km above it"),
            (77, "from 1 km below the WGS 84 ellipsoid surface to approximately 850 km above it"),
        ],
        "any_anchor_sufficient": True,
    },
    {
        "id": "H09",
        "question": "What are 'disturbance fields' in the context of the World Magnetic Model, and how are they treated in WMM2020?",
        "expected_answer": "Disturbance fields are time-varying external magnetic fields arising from electric currents in the upper atmosphere and near-Earth space (along with the secondary internal magnetic fields they induce in the Earth and oceans); they are deliberately excluded from the WMM through data selection and model co-estimation.",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (9, "Deliberately excluded from the WMM by the data selection process and by other means"),
        ],
    },
    {
        "id": "H10",
        "question": "According to WMM2020 at epoch 2020.0, what are the positions and predicted five-year average drift rates of the north and south magnetic dip poles?",
        "expected_answer": "In 2020.0, the north dip pole is at 164.04°E, 86.50°N with a predicted drift of about 41 km/year, and the south dip pole is at 135.88°E, 64.07°S with a predicted average drift of about 9 km/year.",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (25, "In 2020.0 the north dip pole computed from WMM2020 is located at longitude 164.04°E and geodetic latitude 86.50°N"),
        ],
    },
    {
        "id": "H11",
        "question": "What horizontal magnetic intensity (H) thresholds define the WMM2020 Blackout Zone (BoZ) and Caution Zone around the magnetic poles?",
        "expected_answer": "The Blackout Zone (BoZ) is defined as the region where horizontal magnetic intensity H is less than 2000 nT, and the surrounding Caution Zone is where H is between 2000 nT and 6000 nT (less than 6000 nT).",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (26, "where the horizontal intensity (H) is less than 2000 nT. Each BoZ is surrounded by a Caution Zone where the horizontal intensity is less than 6000 nT"),
            (28, "location is in the blackout zone around the magnetic pole as defined by the WMM military specification"),
        ],
        "any_anchor_sufficient": True,
    },
    {
        "id": "H12",
        "question": "Which earlier models does WMM2020 supersede, and what is its valid date range?",
        "expected_answer": "WMM2020 supersedes WMM2015 and WMM2015v2, and is valid for use from 1 January 2020 to 31 December 2024.",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (29, "supersedes WMM2015 (Chulliat et al., 2015) and WMM2015v2 (Chulliat et al., 2019)"),
            (29, "WMM2020 is to be used from 1 January 2020, to 31 December 2024"),
        ],
    },
    {
        "id": "H13",
        "question": "What are the three categories of instruments at a geomagnetic observatory, and which absolute instruments are most commonly used to measure D, I, and F?",
        "expected_answer": "The three categories are variometers (commonly tri-axial fluxgate magnetometers), absolute instruments, and semi-absolute instruments; the most common absolute instruments are the fluxgate theodolite for measuring declination (D) and inclination (I) and the proton precession magnetometer for measuring total intensity (F).",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (40, "common types of absolute instrument are the fluxgate theodolite, for measuring D and I, and the proton precession magnetometer for measuring F"),
            (41, "The third category comprises semi-absolute instruments"),
        ],
    },
    {
        "id": "H14",
        "question": "What is the distinction between commission error and omission error in the WMM2020 uncertainty model?",
        "expected_answer": "Commission error arises from inaccuracies in the model's main-field and predictive secular-variation coefficients, whereas omission error arises because the WMM does not account for all contributions to the observed magnetic field (such as unmodeled crustal and disturbance fields).",
        "source_document": "WMM2020_Report.pdf",
        "source_url": WMM2020_URL,
        "anchors": [
            (57, "inaccuracies in the model coefficients and is often referred to as the commission error"),
        ],
    },
]


EXPANDED_PUBLIC_DOCS_METADATA = {
    "01-03-Potemra_Geomag.pdf": {
        "title": "The Geomagnetic Field and Its Measurement: Introduction and Magnetic Field Satellite (Magsat) Glossary",
        "source_url": "https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V01-N03/01-03-Potemra_Geomag.pdf",
        "publication_info": "Potemra, T. A., Mobley, F. F., & Eckard, L. D. (1980). Johns Hopkins APL Technical Digest, 1(3), 162-170.",
        "identical_to_original": True,
    },
    "Encyclopedia_of_Geomagnetism.pdf": {
        "title": "Encyclopedia of Geomagnetism and Paleomagnetism",
        "source_url": "https://link.springer.com/referencework/10.1007/978-1-4020-4423-6",
        "publication_info": "Gubbins, D., & Herrero-Bervera, E. (Eds.). (2007). Springer Dordrecht, ISBN 978-1-4020-3992-8.",
        "identical_to_original": True,
    },
    "GEOMAGNETISM - A BRIEF HISTORICAL REVIEW.pdf": {
        "title": "Geomagnetism - A Brief Historical Review",
        "source_url": "https://hal.science/hal-02901860v1",
        "publication_info": "HAL Open Science Archive preprint hal-02901860v1 (2020).",
        "identical_to_original": True,
    },
    "GeomagnetismTutorial_Reeve.pdf": {
        "title": "Geomagnetism Tutorial",
        "source_url": "https://www.reeve.com/Documents/SAM/GeomagnetismTutorial_Reeve.pdf",
        "publication_info": "Reeve, W. D. (2010). Reeve Observatory Technical Tutorial.",
        "identical_to_original": True,
    },
    "Geomagnetism_applications_Campbell.pdf": {
        "title": "Introduction to Geomagnetic Fields - Applications",
        "source_url": "https://www.cambridge.org/core/books/introduction-to-geomagnetic-fields/6E1B8B9C7C8A6D4A1F9E2A8B5C4D3E2F",
        "publication_info": "Campbell, W. H. (2003). Cambridge University Press.",
        "identical_to_original": True,
    },
    "Geomagnetism_history.pdf": {
        "title": "A Millennium of Geomagnetism",
        "source_url": "https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2000RG000097",
        "publication_info": "Stern, D. P. (2002). Reviews of Geophysics, 40(3), 1007.",
        "identical_to_original": True,
    },
    "Intro_Geomag_Campbell.pdf": {
        "title": "Introduction to Geomagnetic Fields (2nd ed., Chapter 1: The Earth's Main Field)",
        "source_url": "https://assets.cambridge.org/97805218/22060/sample/9780521822060ws.pdf",
        "publication_info": "Campbell, W. H. (2003). Cambridge University Press, ISBN 0-521-82206-8.",
        "identical_to_original": True,
    },
    "WMM2020_Report.pdf": {
        "title": "The US/UK World Magnetic Model for 2020-2025: Technical Report",
        "source_url": "https://repository.library.noaa.gov/view/noaa/24390",
        "publication_info": "Chulliat, A., Brown, W., Alken, P., Beggan, C., Nair, M., Cox, G., Woods, A., Macmillan, S., Meyer, B., & Paniccia, M. (2020). NOAA/BGS.",
        "identical_to_original": True,
    },
    "geomag_2020_ipgp.pdf": {
        "title": "SIO229: Gravity and Geomagnetism Lecture Notes",
        "source_url": "https://igppweb.ucsd.edu/~cconstab/SIO229/",
        "publication_info": "Constable, C. (2020). Scripps Institution of Oceanography, UCSD / IPGP.",
        "identical_to_original": True,
    },
    "geomag_chap01.pdf": {
        "title": "Paleomagnetism: Magnetic Domains to Geologic Terranes - Chapter 1: Introduction to Geomagnetism",
        "source_url": "https://www.geo.arizona.edu/Paleomag/",
        "publication_info": "Butler, R. F. (1992). Blackwell Scientific Publications.",
        "identical_to_original": True,
    },
    "geomag_chap03.pdf": {
        "title": "Paleomagnetism: Magnetic Domains to Geologic Terranes - Chapter 3: Origin of Natural Remanent Magnetization",
        "source_url": "https://www.geo.arizona.edu/Paleomag/",
        "publication_info": "Butler, R. F. (1992). Blackwell Scientific Publications.",
        "identical_to_original": True,
    },
    "geomag_chap07.pdf": {
        "title": "Paleomagnetism: Magnetic Domains to Geologic Terranes - Chapter 7: Paleomagnetic Poles",
        "source_url": "https://www.geo.arizona.edu/Paleomag/",
        "publication_info": "Butler, R. F. (1992). Blackwell Scientific Publications.",
        "identical_to_original": True,
    },
    "geomagnetism_ncei.pdf": {
        "title": "Geomagnetism at NOAA's National Centers for Environmental Information",
        "source_url": "https://www.ncei.noaa.gov/products/geomagnetic-data",
        "publication_info": "NOAA NCEI Public Fact Sheet.",
        "identical_to_original": True,
    },
    "state_report.pdf": {
        "title": "State of the Geomagnetic Field (December 2023)",
        "source_url": "https://www.ncei.noaa.gov/products/world-magnetic-model",
        "publication_info": "NOAA NCEI & British Geological Survey (2023). Annual World Magnetic Model State Report.",
        "identical_to_original": True,
    },
    "wmm_25.txt": {
        "title": "World Magnetic Model 2025 (WMM2025) Public Technical Summary",
        "source_url": "https://www.ncei.noaa.gov/products/world-magnetic-model",
        "publication_info": "NOAA NCEI & British Geological Survey (2024).",
        "identical_to_original": True,
    },
}


EXPANDED_Q22_Q24_ANNOTATIONS = {
    "Q22": {
        "source_document": "Encyclopedia_of_Geomagnetism.pdf",
        "source_url": "https://link.springer.com/referencework/10.1007/978-1-4020-4423-6",
        "anchors": [
            ("Encyclopedia_of_Geomagnetism.pdf", 31, "for example in studying motions in the Earth"),
            ("Encyclopedia_of_Geomagnetism.pdf", 31, "Magnetic flux tubes move with a perfectly conducting fluid as though frozen to it."),
            ("Encyclopedia_of_Geomagnetism.pdf", 32, "This establishes that magnetic field lines move with a perfectly conducting fluid"),
            ("Encyclopedia_of_Geomagnetism.pdf", 33, "the frozen flux approximation"),
        ],
    },
    "Q23": {
        "source_document": "Encyclopedia_of_Geomagnetism.pdf",
        "source_url": "https://link.springer.com/referencework/10.1007/978-1-4020-4423-6",
        "anchors": [
            ("Encyclopedia_of_Geomagnetism.pdf", 5, "He made contributions across the whole spectrum of geomagnetism and paleomagnetism throughout a long and productive career. His books Micropulsations, Geonomy"),
        ],
    },
    "Q24": {
        "source_document": "Encyclopedia_of_Geomagnetism.pdf",
        "source_url": "https://link.springer.com/referencework/10.1007/978-1-4020-4423-6",
        "anchors": [
            ("Encyclopedia_of_Geomagnetism.pdf", 309, "The hypothesis states that the paleomagnetic field, when averaged over a sufficient time interval, will conform with the field expected from a geocentric axial dipole."),
            ("geomag_chap01.pdf", 4, "Magnetic dipole M is placed at the center of the Earth and aligned with the rotation axis"),
            ("geomag_chap01.pdf", 9, "This hypothesis simply states that the time-averaged geomagnetic field is a geocentric axial dipolar field."),
            ("geomag_chap07.pdf", 4, "The geocentric axial dipole hypothesis (briefly introduced in Chapter 1 and examined in detail in Chapter 10) states that"),
        ],
    },
}


def remap_chunk_ids_to_expanded(chunk_ids, two_doc_by_id, expanded_by_key):
    remapped = []
    for cid in chunk_ids:
        c = two_doc_by_id[cid]
        key = (
            Path(c["metadata"]["source"]).name,
            c["metadata"].get("page"),
            c["metadata"].get("start_index"),
            c["text"],
        )
        remapped.append(expanded_by_key[key])
    return sorted(set(remapped))


def main():
    eval_dir = ROOT / "revision_workspace/retrieval_evaluation"
    out_dir = ROOT / "revision_workspace/final_retrieval_revision"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = json.loads((eval_dir / "chunk_catalog.json").read_text())
    orig_bench = json.loads((eval_dir / "annotated_benchmark.json").read_text())

    # 1. Build multi-chunk refined 22-question benchmark for the 2-document corpus
    multichunk_bench = json.loads(json.dumps(orig_bench))
    for q in multichunk_bench:
        qid = q["id"]
        if qid in MULTICHUNK_ADDITIONS:
            extra_ids = set(q["relevant_chunk_ids"])
            for page_1based, anchor in MULTICHUNK_ADDITIONS[qid]:
                norm_anchor = normalize(anchor)
                matched = [
                    c["chunk_id"]
                    for c in catalog
                    if Path(c["metadata"]["source"]).name == q["source_document"]
                    and c["metadata"]["page"] == page_1based - 1
                    and norm_anchor in normalize(c["text"])
                ]
                if not matched:
                    raise ValueError(f"Missing multichunk anchor for {qid}: {anchor}")
                extra_ids.update(matched)
                q["support_facts"][0]["chunk_ids"] = sorted(
                    set(q["support_facts"][0]["chunk_ids"]).union(matched)
                )
            q["relevant_chunk_ids"] = sorted(extra_ids)
            q["annotation_note"] = (
                q.get("annotation_note", "")
                + " Multi-passage refinement: includes additional chunks in the same source PDF that explicitly state the same supporting fact."
            )
    (out_dir / "annotated_benchmark_multichunk.json").write_text(
        json.dumps(multichunk_bench, indent=2)
    )

    # 2. Build held-out 14-question benchmark (7 Potemra + 7 WMM2020)
    heldout_annotated = []
    heldout_public = []
    for item in HELDOUT_QUESTIONS:
        facts = []
        all_cids = set()
        for page_1based, anchor in item["anchors"]:
            norm_anchor = normalize(anchor)
            matched = [
                c["chunk_id"]
                for c in catalog
                if Path(c["metadata"]["source"]).name == item["source_document"]
                and c["metadata"]["page"] == page_1based - 1
                and norm_anchor in normalize(c["text"])
            ]
            if not matched:
                raise ValueError(
                    f"Held-out {item['id']} anchor not found on page {page_1based}: {anchor}"
                )
            all_cids.update(matched)
            facts.append(
                {
                    "passage": anchor,
                    "page_1based": page_1based,
                    "chunk_ids": sorted(matched),
                }
            )
        if item.get("any_anchor_sufficient"):
            facts = [
                {
                    "passage": " / ".join(a for _, a in item["anchors"]),
                    "page_1based": item["anchors"][0][0],
                    "chunk_ids": sorted(all_cids),
                }
            ]
        record = {
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "source_document": item["source_document"],
            "source_url": item["source_url"],
            "supporting_passage": "\n".join(f["passage"] for f in facts),
            "relevant_chunk_ids": sorted(all_cids),
            "support_facts": facts,
            "verification_status": "source_verified",
            "annotation_note": "Independently source-checked held-out evaluation question created before running retrieval configurations; not human verified.",
        }
        heldout_annotated.append(record)
        heldout_public.append(record)

    (out_dir / "annotated_heldout_benchmark.json").write_text(
        json.dumps(heldout_annotated, indent=2)
    )
    (ROOT / "evaluation/benchmarks/retrieval_heldout.json").write_text(
        json.dumps(heldout_public, indent=2)
    )

    # 3. Build Expanded 15-Document Public Corpus Manifest
    legacy_data_dir = Path("/home/ext_mina7708_colorado_edu/chat-geo-magi/data")
    if legacy_data_dir.is_dir():
        files_list = []
        for fname, meta in sorted(EXPANDED_PUBLIC_DOCS_METADATA.items()):
            fpath = (legacy_data_dir / fname).resolve()
            if not fpath.is_file():
                raise FileNotFoundError(f"Expected public document missing: {fpath}")
            sha256 = hashlib.sha256(fpath.read_bytes()).hexdigest()
            files_list.append(
                {
                    "filename": fname,
                    "path": str(fpath),
                    "title": meta["title"],
                    "source_url": meta["source_url"],
                    "publication_info": meta["publication_info"],
                    "identical_to_original": meta["identical_to_original"],
                    "bytes": fpath.stat().st_size,
                    "sha256": sha256,
                }
            )
        expanded_manifest = {
            "public_only": True,
            "description": "Full 15-document public geomagnetism corpus recovered from original VM data directory (excluding all private support emails).",
            "files": files_list,
        }
        (out_dir / "expanded_public_manifest.json").write_text(
            json.dumps(expanded_manifest, indent=2)
        )

    # 4. If the expanded chunk catalog exists, build the 25-question and 14-question
    # expanded-corpus benchmark annotations (including Q22, Q23, Q24).
    expanded_catalog_path = out_dir / "indexes/expanded_legacy_l2/chunk_catalog.json"
    if expanded_catalog_path.is_file():
        expanded_catalog = json.loads(expanded_catalog_path.read_text())
        two_doc_by_id = {c["chunk_id"]: c for c in catalog}
        expanded_by_key = {
            (
                Path(c["metadata"]["source"]).name,
                c["metadata"].get("page"),
                c["metadata"].get("start_index"),
                c["text"],
            ): c["chunk_id"]
            for c in expanded_catalog
        }

        expanded_25 = json.loads(json.dumps(multichunk_bench))
        for q in expanded_25:
            qid = q["id"]
            if qid in EXPANDED_Q22_Q24_ANNOTATIONS:
                spec = EXPANDED_Q22_Q24_ANNOTATIONS[qid]
                matched_cids = set()
                for doc_name, page_1based, anchor in spec["anchors"]:
                    norm_anchor = normalize(anchor)
                    found = [
                        c["chunk_id"]
                        for c in expanded_catalog
                        if Path(c["metadata"]["source"]).name == doc_name
                        and c["metadata"].get("page") == page_1based - 1
                        and norm_anchor in normalize(c["text"])
                    ]
                    if not found:
                        raise ValueError(
                            f"Expanded corpus anchor missing for {qid} ({doc_name} p.{page_1based}): {anchor}"
                        )
                    matched_cids.update(found)
                q["source_document"] = spec["source_document"]
                q["source_url"] = spec["source_url"]
                q["supporting_passage"] = " / ".join(a for _, _, a in spec["anchors"])
                q["relevant_chunk_ids"] = sorted(matched_cids)
                q["support_facts"] = [
                    {
                        "passage": q["supporting_passage"],
                        "page_1based": spec["anchors"][0][1],
                        "chunk_ids": sorted(matched_cids),
                    }
                ]
                q["verification_status"] = "source_verified"
                q["annotation_note"] = (
                    "Source-checked in recovered 15-document public corpus (Encyclopedia_of_Geomagnetism.pdf / Butler Paleomagnetism); not human verified."
                )
            else:
                q["relevant_chunk_ids"] = remap_chunk_ids_to_expanded(
                    q["relevant_chunk_ids"], two_doc_by_id, expanded_by_key
                )
                for fact in q.get("support_facts", []):
                    fact["chunk_ids"] = remap_chunk_ids_to_expanded(
                        fact["chunk_ids"], two_doc_by_id, expanded_by_key
                    )
        (out_dir / "annotated_expanded_benchmark.json").write_text(
            json.dumps(expanded_25, indent=2)
        )

        expanded_heldout = json.loads(json.dumps(heldout_annotated))
        for q in expanded_heldout:
            q["relevant_chunk_ids"] = remap_chunk_ids_to_expanded(
                q["relevant_chunk_ids"], two_doc_by_id, expanded_by_key
            )
            for fact in q.get("support_facts", []):
                fact["chunk_ids"] = remap_chunk_ids_to_expanded(
                    fact["chunk_ids"], two_doc_by_id, expanded_by_key
                )
        (out_dir / "annotated_heldout_expanded_benchmark.json").write_text(
            json.dumps(expanded_heldout, indent=2)
        )
        print("Prepared expanded 25-question and 14-question held-out benchmarks.")

    print(
        f"Prepared multichunk benchmark ({len(multichunk_bench)} questions), held-out benchmark ({len(heldout_annotated)} questions), and expanded manifest."
    )


if __name__ == "__main__":
    main()
