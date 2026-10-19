"""Retrieve public reading copies and bibliographic records, without saving PDFs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote

import pymupdf as fitz
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "v2/paper/materical/literature_sources"
PAPERS = [
    ("heterogeneity", "Dealing with spatial heterogeneity", "10.1007/s10040-004-0432-3", None),
    ("idw", "An adaptive inverse-distance weighting spatial interpolation technique", "10.1016/j.cageo.2007.07.010", None),
    ("geocalibration", "Impact of geostatistical reconstruction approaches on model calibration for flow in highly heterogeneous aquifers", "10.1007/s00477-020-01865-2", None),
    ("gstools", "GSTools v1.3: a toolbox for geostatistical modelling in Python", "10.5194/gmd-15-3161-2022", None),
    ("spectral", "On the Spectral Bias of Neural Networks", None, "1806.08734"),
    ("fourier", "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", None, "2006.10739"),
    ("siren", "Implicit Neural Representations with Periodic Activation Functions", None, "2006.09661"),
    ("pinn", "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", "10.1016/j.jcp.2018.10.045", "1711.10561"),
    ("fdpinn", "FD-PINN: FREQUENCY DOMAIN PHYSICS-INFORMED NEURAL NETWORK", "10.6052/0459-1879-23-169", None),
    ("gradient", "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks", "10.1137/20M1318043", "2001.04536"),
    ("sapinn", "Self-adaptive physics-informed neural networks", "10.1016/j.jcp.2022.111722", "2009.04544"),
    ("rba", "Residual-based attention in physics-informed neural networks", "10.1016/j.cma.2024.116805", "2307.00379"),
    ("annulus", "用于两相流环空压力预测的自适应物理信息神经网络模型", "10.7623/syxb202303012", None),
    ("scipy", "SciPy 1.0: fundamental algorithms for scientific computing in Python", "10.1038/s41592-019-0686-2", "1907.10121"),
    ("ntk", "When and why PINNs fail to train: A neural tangent kernel perspective", "10.1016/j.jcp.2021.110768", "2007.14527"),
    ("failure", "Characterizing possible failure modes in physics-informed neural networks", None, "2109.01050"),
    ("guide", "An expert's guide to training physics-informed neural networks", "10.48550/arXiv.2308.08468", "2308.08468"),
    ("review", "Scientific Machine Learning Through Physics-Informed Neural Networks: Where we are and What's Next", "10.1007/s10915-022-01939-z", "2201.05624"),
    ("sampling", "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks", "10.1016/j.cma.2022.115671", "2207.10289"),
    ("deepxde", "DeepXDE: A Deep Learning Library for Solving Differential Equations", "10.1137/19M1274067", "1907.04502"),
    ("deeponet", "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators", "10.1038/s42256-021-00302-5", "1910.03193"),
    ("fno", "Fourier Neural Operator for Parametric Partial Differential Equations", None, "2010.08895"),
    ("deepsdf", "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation", None, "1901.05103"),
    ("nerf", "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", "10.1007/978-3-030-58452-8_24", "2003.08934"),
    ("bacon", "BACON: Band-limited Coordinate Networks for Multiscale Scene Representation", None, "2112.04645"),
    ("activation", "Adaptive activation functions accelerate convergence in deep and physics-informed neural networks", "10.1016/j.jcp.2019.109136", "1906.01170"),
    ("dgm", "DGM: A deep learning algorithm for solving partial differential equations", "10.1016/j.jcp.2018.08.029", "1708.07469"),
    ("ritz", "The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems", "10.1007/s40304-018-0127-z", "1710.00211"),
    ("eigenvector", "On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks", "10.1016/j.cma.2021.113938", "2012.10047"),
    ("gpinn", "Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems", "10.1016/j.cma.2022.114823", "2111.02801"),
    ("causal", "Respecting causality is all you need for training physics-informed neural networks", None, "2203.07404"),
    ("hard", "Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks", "10.1016/j.cma.2021.114333", "2104.08426"),
    ("localactivation", "Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks", "10.1098/rspa.2020.0334", "1909.12228"),
    ("wight", "Solving Allen-Cahn and Cahn-Hilliard Equations using the Adaptive Physics Informed Neural Networks", None, "2007.04542"),
]


def fetch(url):
    response = requests.get(url, timeout=35, headers={"User-Agent": "ManuscriptReferenceCheck/1.0 (public academic reading)"})
    response.raise_for_status()
    return response


def extract(response):
    if response.content.startswith(b"%PDF"):
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return "\n\n".join(f"[Page {i + 1}]\n" + p.get_text() for i, p in enumerate(doc))
    soup = BeautifulSoup(response.content, "html.parser")
    article = soup.select_one("article") or soup.select_one("main") or soup
    for item in article.select("script, style, nav, footer, header"):
        item.decompose()
    return article.get_text("\n", strip=True)


def process(paper):
    key, title, doi, arxiv = paper
    path = OUT / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("text_file") and record.get("doi") == doi:
            return record
    record = {"key": key, "expected_title": title, "doi": doi, "arxiv": arxiv,
              "accessed": datetime.now(timezone.utc).isoformat(), "errors": []}
    if doi and not doi.startswith("10.48550"):
        try:
            record["crossref"] = fetch("https://api.crossref.org/works/" + quote(doi, safe="")).json()["message"]
        except Exception as e:
            record["errors"].append(f"Crossref: {e}")
    urls = []
    if arxiv:
        try:
            soup = BeautifulSoup(fetch("https://arxiv.org/abs/" + arxiv).content, "html.parser")
            record["arxiv_metadata"] = {name: [m.get("content") for m in soup.select(f'meta[name="{name}"]')]
                                       for name in ("citation_title", "citation_author", "citation_date", "citation_doi", "citation_pdf_url")}
            journal = soup.select_one(".jref")
            record["journal_ref"] = journal.get_text(" ", strip=True) if journal else None
        except Exception as e:
            record["errors"].append(f"arXiv metadata: {e}")
        urls.extend(["https://arxiv.org/html/" + arxiv, "https://arxiv.org/pdf/" + arxiv])
    if key == "gstools":
        urls.insert(0, "https://gmd.copernicus.org/articles/15/3161/2022/")
    if key in ("fdpinn", "annulus"):
        source = ROOT / "tmp/pdfs/chinese_core_references" / ("fd_pinn_extracted.txt" if key == "fdpinn" else "syxb_extracted.txt")
        record["text_file"] = str(source.relative_to(ROOT))
        record["reading_source"] = str(source.relative_to(ROOT))
    elif not arxiv and doi:
        urls.extend(link["URL"] for link in record.get("crossref", {}).get("link", []) if "pdf" in link.get("content-type", ""))
        urls.append("https://doi.org/" + doi)
    if not record.get("text_file"):
        for url in urls:
            try:
                response = fetch(url)
                text = extract(response)
                if "This is a preview of subscription content" in text:
                    raise ValueError("Subscription preview, not full text")
                if len(text) < 12000:
                    raise ValueError(f"Only {len(text)} characters; not accepted as full text")
                target = OUT / f"{key}.txt"
                target.write_text(text)
                record.update(text_file=str(target.relative_to(ROOT)), reading_source=response.url,
                              text_chars=len(text), is_pdf_reading=response.content.startswith(b"%PDF"))
                break
            except Exception as e:
                record["errors"].append(f"{url}: {e}")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = [p for p in PAPERS if not args.keys or p[0] in args.keys]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for record in pool.map(process, selected):
            print(json.dumps({"key": record["key"], "full_text": bool(record.get("text_file")),
                              "actual_title": record.get("arxiv_metadata", {}).get("citation_title") or record.get("crossref", {}).get("title"),
                              "errors": record["errors"][-1:]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
