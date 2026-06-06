#!/usr/bin/env python3
"""Build anonymized COLM-format OpenReview PDFs from the paper drafts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "colm_final" / "paper" / "template"
PAPER = ROOT / "paper"


PAPERS = {
    "latentwire": {
        "dir": PAPER / "latentwire",
        "title": "When Byte-Scale Model Communication Should Be Text",
        "subtitle": "A Controlled Negative for Discrete No-Text Packets",
        "out": ROOT / "openreview_latentwire.pdf",
    },
    "channel_set": {
        "dir": PAPER / "channel_set",
        "title": "Channel Sets Drift During Long Reasoning",
        "subtitle": "A Measurement Regime for Static W4A16 Protection",
        "out": ROOT / "openreview_channel_set.pdf",
    },
}


CODE_REPLACEMENTS = {
    "KILL_UTILITY_IS_IDENTITY": "utility is identity leakage",
    "L-IB1": "the privacy-bottleneck escape",
    "L-B1": "the damage-avoidance escape",
    "L-Q1": "the receiver-query escape",
    "L-PC5/L-C2": "rerank/fuser",
    "L-PC5": "rerank",
    "L-C2": "fuser",
    "C-A1": "the adaptive clip candidate",
    "C_A1": "the adaptive clip candidate",
    "C-F": "the survival-core candidate",
    "C_U1": "the router audit",
    "C-S1": "the clean-denominator audit",
    "C_S1": "the clean-denominator audit",
    "C-Y5": "the defense-bundle audit",
    "C_Y5": "the defense-bundle audit",
    "LW-HO": "Held-out",
}


LATENTWIRE_INSERT = (
    "Adjacent calibrated-routing and structured-communication systems, including "
    "UCCI [@ucci2026], DarkForest [@darkforest2026], structured message passing "
    "[@smp2013], and CU-HLM [@cu_hlm2026], motivate the same hard-baseline "
    "discipline: a packet must beat the visible or routed evidence that an "
    "ordinary system could transmit directly."
)


ERROR_MODEL_OLD = (
    "The downstream relevance of set-leaving follows from a simple error model. "
    "A channel inside the protected set incurs the protected per-channel error "
    "`epsilon_p`; a high-risk channel that has left it incurs the unprotected "
    "error `epsilon_q >> epsilon_p`. Writing `L(t0,t)` for the fraction of later "
    "high-risk channels outside the set chosen at `t0`, the expected excess "
    "quantization error behaves as `E[err(t)] ~= K { epsilon_p (1 - L(t0,t)) + "
    "epsilon_q L(t0,t) }`, which is increasing in `L`. Set-leaving is therefore "
    "not merely a membership statistic: under this model the measured `L` of "
    "`0.53`-`0.67` places a large fraction of later high-risk channels at the "
    "unprotected error rate. The model assumes leaving channels carry comparable "
    "risk to those retained; we treat the per-channel error attribution itself "
    "as the parked C-A1 question and claim no confirmed accuracy delta here."
)


ERROR_MODEL_NEW = (
    "The downstream relevance of set-leaving follows from a simple error model. "
    "A channel inside the protected set incurs protected per-channel error "
    "$\\varepsilon_p$; a high-risk channel that has left it incurs unprotected "
    "error $\\varepsilon_q \\gg \\varepsilon_p$. Let $L=L(t_0,t)$ be the "
    "fraction of later high-risk channels outside the set chosen at $t_0$. Then\n\n"
    "$$E[\\mathrm{err}(t)] \\approx K\\{\\varepsilon_p(1-L) + "
    "\\varepsilon_q L\\}.$$\n\n"
    "which is increasing in $L$. Set-leaving is therefore not merely a "
    "membership statistic: under this model the measured $L$ of $0.53$--$0.67$ "
    "places a large fraction of later high-risk channels at the unprotected "
    "error rate. The model assumes leaving channels carry comparable risk to "
    "those retained; we treat the per-channel error attribution itself as the "
    "parked adaptive-method question and claim no confirmed accuracy delta here."
)


def run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def ensure_template_files() -> None:
    for name in ["colm2026_conference.sty", "colm2026_conference.bst"]:
        shutil.copy2(TEMPLATE / name, PAPER / name)


def extract_sections(markdown: str) -> tuple[str, str, str, str]:
    lines = markdown.splitlines()
    title = lines[0].removeprefix("# ").strip()
    subtitle = ""
    abstract_start = None
    abstract_end = None
    for idx, line in enumerate(lines):
        if idx > 0 and line.startswith("## ") and not subtitle:
            subtitle = line.removeprefix("## ").strip()
        if line.strip() == "## Abstract":
            abstract_start = idx + 1
        elif abstract_start is not None and idx > abstract_start and line.startswith("## "):
            abstract_end = idx
            break
    if abstract_start is None or abstract_end is None:
        raise ValueError("could not locate abstract block")
    abstract = "\n".join(lines[abstract_start:abstract_end]).strip()
    body = "\n".join(lines[abstract_end:])
    return title, subtitle, abstract, body


def scrub_markdown(text: str, paper_id: str) -> str:
    text = text.replace(ERROR_MODEL_OLD, ERROR_MODEL_NEW)
    for old, new in CODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.replace(
        "All paper-finalization work is artifact-only. No new experiment, GPU run, MPS live forward, confirmation split access, or killed-method rerun is used here. ",
        "",
    )
    text = text.replace(
        "The detailed provenance is in `tables/provenance.md`.",
        "Claim provenance is summarized in the text and figures.",
    )
    text = text.replace(
        "The detailed provenance is in `tables/provenance.md`; the final regime checklist is in `tables/regime_checklist.md`.",
        "The final regime checklist is summarized in the text below.",
    )
    text = text.replace(
        "The full provenance table is in `tables/provenance.md`.",
        "The full audit provenance is retained internally; the submitted paper reports only the claim-bearing summaries.",
    )
    if paper_id == "latentwire" and LATENTWIRE_INSERT not in text:
        marker = (
            "Privacy-preserving semantic communication and adaptive text anonymization motivate the privacy frontier "
            "[@ibal2023; @adaptive_anonymization2026]."
        )
        text = text.replace(marker, LATENTWIRE_INSERT + "\n\n" + marker)
    text = text.replace(
        "Figure 1: The source-score signal largely disappears once receiver evidence is included.",
        "Figure 1: Powered cached support: the source-score signal largely disappears once receiver evidence is included.",
    )
    text = text.replace(
        "Figure 3: The exact visible public signature reaches `1.000` accuracy, while the learned matched packet reaches `0.775`. Controls remain at chance.",
        "Figure 3: Cached exact-evidence support: the exact visible public signature reaches `1.000` accuracy, while the learned matched packet reaches `0.775`. Controls remain at chance.",
    )
    text = text.replace(
        "Figure 4: L-IB1 cannot move off the utility/leakage identity diagonal enough to preserve utility while hiding source/evidence structure.",
        "Figure 4: Cached privacy-bottleneck blocker: the packet cannot move off the utility/leakage identity diagonal enough to preserve utility while hiding source/evidence structure.",
    )
    text = text.replace("C2C/KVComm smokes", "C2C/KVComm local checks")
    text = text.replace("Gold-aware", "Answer-aware")
    return text


def latex_from_markdown(markdown: str, cwd: Path) -> str:
    proc = subprocess.run(
        ["pandoc", "--from", "markdown", "--to", "latex", "--natbib"],
        input=markdown,
        cwd=cwd,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def cleanup_latex(body: str) -> str:
    body = re.sub(
        r"\\hypertarget\{[^{}]+\}\{%\n(\\(?:sub)*section\{[^{}]+\}\\label\{[^{}]+\})\n\}",
        r"\1",
        body,
    )
    body = re.sub(r"\\texttt\{([-+0-9.,/\\ \[\]]+)\}", lambda m: "\\(" + m.group(1).replace("\\ ", " ") + "\\)", body)
    body = body.replace("\\texttt{utility\\ is\\ identity\\ leakage}", "utility is identity leakage")
    body = body.replace("\\_", "-")
    body = re.sub(r"\\begin\{longtable\}", "\\\\begingroup\\\\small\n\\\\begin{longtable}", body)
    body = body.replace("\\end{longtable}", "\\end{longtable}\n\\endgroup")
    return body


def abstract_to_latex(markdown: str, cwd: Path) -> str:
    text = latex_from_markdown(scrub_markdown(markdown, "abstract"), cwd).strip()
    text = re.sub(r"\\texttt\{([-+0-9.,/\\ \[\]]+)\}", lambda m: "\\(" + m.group(1).replace("\\ ", " ") + "\\)", text)
    text = text.replace("\\texttt{utility\\ is\\ identity\\ leakage}", "utility is identity leakage")
    return text


def wrapper(title: str, subtitle: str, abstract: str, body: str) -> str:
    return rf"""
\documentclass{{article}}
\usepackage[submission]{{../colm2026_conference}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{microtype}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{caption}}
\usepackage{{lineno}}

\definecolor{{darkblue}}{{rgb}}{{0, 0, 0.5}}
\hypersetup{{
  colorlinks=true,
  citecolor=darkblue,
  linkcolor=darkblue,
  urlcolor=darkblue,
  pdfauthor={{Anonymous}},
  pdftitle={{{title}}},
  pdfsubject={{COLM 2026 submission}},
  pdfkeywords={{}},
  pdfcreator={{LaTeX}},
  pdfproducer={{}}
}}
\urlstyle{{same}}
\graphicspath{{{{figures/}}}}
\setkeys{{Gin}}{{width=0.92\linewidth,height=0.27\textheight,keepaspectratio}}
\setcounter{{secnumdepth}}{{2}}
\newcommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}

\title{{{title}\\\large {subtitle}}}
\author{{Anonymous Authors}}

\begin{{document}}
\ifcolmsubmission
\linenumbers
\fi
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

{body}

\bibliographystyle{{../colm2026_conference}}
\bibliography{{../references}}

\end{{document}}
""".strip() + "\n"


def build_one(paper_id: str, spec: dict[str, Path | str]) -> None:
    paper_dir = spec["dir"]
    markdown = (paper_dir / "draft.md").read_text(encoding="utf-8")
    title, subtitle, abstract_md, body_md = extract_sections(markdown)
    body_md = scrub_markdown(body_md, paper_id)
    abstract_md = scrub_markdown(abstract_md, paper_id)
    body = cleanup_latex(latex_from_markdown(body_md, paper_dir))
    abstract = abstract_to_latex(abstract_md, paper_dir)
    tex = wrapper(title, subtitle, abstract, body)
    (paper_dir / "main.tex").write_text(tex, encoding="utf-8")
    for artifact in paper_dir.glob("main.*"):
        if artifact.suffix not in {".tex", ".pdf"}:
            artifact.unlink()
    run(["latexmk", "-pdf", "-gg", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], paper_dir)
    shutil.copy2(paper_dir / "main.pdf", spec["out"])


def main() -> int:
    ensure_template_files()
    for paper_id, spec in PAPERS.items():
        build_one(paper_id, spec)
        print(f"wrote {Path(spec['out']).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
