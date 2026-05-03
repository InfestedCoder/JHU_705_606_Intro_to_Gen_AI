"""Render the Optimization Log Copilot pipeline diagram.

Generates ``architecture.png`` from a Graphviz description of the four-stage
pipeline used in the final paper:

  1. Parsing - raw Rose solver logs (JSON + stdout) -> SolverLogFeatures
  2. Feature engineering - 25 numerical features (log1p + StandardScaler)
  3. Bottleneck detection - heuristic rules + PyTorch MLP, both producing
     a 5-label multi-label vector
  4. Diagnostic generation - label-conditioned retrieval over a curated
     knowledge base, fed to Llama-3.2-1B-Instruct via the chat template

A 30-run hand-reviewed gold subset evaluates the bottleneck labels and the
generated diagnostics.
"""
from graphviz import Digraph
from graphviz.backend.execute import ExecutableNotFound

try:
    from IPython.display import display
except ImportError:
    display = None


dot = Digraph("solver_log_copilot_pipeline", format="png")
dot.attr(rankdir="TB", splines="true", nodesep="0.55", ranksep="0.65")
dot.attr("node", shape="box", style="rounded", fontsize="13", fontname="Helvetica")
dot.attr("edge", arrowsize="0.8", fontname="Helvetica", fontsize="10")
dot.graph_attr["dpi"] = "300"

# Color palette
INPUT_ATTR  = {"fillcolor": "#d5e8d4", "style": "filled,rounded"}   # green - inputs
STAGE_ATTR  = {"fillcolor": "#dae8fc", "style": "filled,rounded"}   # blue - stages
OUTPUT_ATTR = {"fillcolor": "#fff2cc", "style": "filled,rounded"}   # yellow - outputs
AUX_ATTR    = {"fillcolor": "#e8e8e8", "style": "filled,rounded"}   # gray - auxiliary
EVAL_ATTR   = {"fillcolor": "#f8cecc", "style": "filled,rounded"}   # pink - evaluation

# ------------------------------------------------------------------
# Nodes
# ------------------------------------------------------------------
# Stage 0 - inputs
dot.node("A", "Raw Solver Logs\n(JSON metrics + stdout)", **INPUT_ATTR)

# Stage 1 - parsing
dot.node("B", "Parser\n(parse_rose_logs_v2.py)", **STAGE_ATTR)

# Stage 2 - feature representation
dot.node("C", "SolverLogFeatures\n(25 numerical features,\nlog1p + StandardScaler)", **STAGE_ATTR)

# Stage 3 - bottleneck detection (two parallel paths)
dot.node("D1", "Heuristic Rules\n(5 thresholded rules)", **STAGE_ATTR)
dot.node("D2", "PyTorch Classifier\n(MLP, 6,149 params,\nBCE + WeightedRandomSampler)", **STAGE_ATTR)
dot.node("L",  "Bottleneck Labels\n(weak_root_lp, excessive_branching,\nineffective_cuts, degeneracy,\npresolve_weakness)", **OUTPUT_ATTR)

# Stage 4 - RAG diagnostic generation
dot.node("KB", "Knowledge Base\n(12 docs, 6 topics)", **AUX_ATTR)
dot.node("E",  "SentenceTransformer\n(all-MiniLM-L6-v2)", **STAGE_ATTR)
dot.node("R",  "Label-Conditioned\nRetrieval (top-k)", **STAGE_ATTR)
dot.node("F",  "Llama-3.2-1B-Instruct\n(chat template)", **STAGE_ATTR)
dot.node("G",  "Diagnostic Report\n(one paragraph)", **OUTPUT_ATTR)

# Evaluation (orthogonal)
dot.node("EV", "30-run Hand-Reviewed\nGold Subset", **EVAL_ATTR)

# ------------------------------------------------------------------
# Edges - main pipeline
# ------------------------------------------------------------------
dot.edge("A", "B")
dot.edge("B", "C")

# Bottleneck detection: features feed both paths; both paths produce labels
dot.edge("C", "D1")
dot.edge("C", "D2")
dot.edge("D1", "L")
dot.edge("D2", "L")

# RAG flow - keep the diagram to the canonical retrieval pipeline; the
# prose describes the additional wires (run summary + anchor labels also
# fed to the LLM as part of the user message).
dot.edge("KB", "E")
dot.edge("E", "R")
dot.edge("L", "R", label="filter by topic", style="dashed")
dot.edge("R", "F")
dot.edge("F", "G")

# Evaluation - dashed gray on the side
dot.edge("EV", "L", style="dashed", color="#888888", fontcolor="#888888",
         label="evaluates")
dot.edge("EV", "G", style="dashed", color="#888888", fontcolor="#888888")

# ------------------------------------------------------------------
# Stage labels (left margin)
# ------------------------------------------------------------------
def stage_label(name, text):
    dot.node(name, text, shape="plaintext", fontsize="10",
             fontcolor="gray", fontname="Helvetica-Oblique")


stage_label("L0", "Input")
stage_label("L1", "Stage 1: Parse")
stage_label("L2", "Stage 2: Features")
stage_label("L3", "Stage 3: Detect")
stage_label("L4", "Stage 4: Diagnose")
stage_label("L5", "Output")

# Invisible chain to vertically order the stage labels
dot.edge("L0", "L1", style="invis")
dot.edge("L1", "L2", style="invis")
dot.edge("L2", "L3", style="invis")
dot.edge("L3", "L4", style="invis")
dot.edge("L4", "L5", style="invis")

# Align each stage label horizontally with its content
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("L0"); s.node("A")
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("L1"); s.node("B")
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("L2"); s.node("C")
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("L3"); s.node("D1"); s.node("D2"); s.node("KB"); s.node("E")
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("L4"); s.node("L"); s.node("R"); s.node("EV")
with dot.subgraph() as s:
    s.attr(rank="same"); s.node("L5"); s.node("F"); s.node("G")

# ------------------------------------------------------------------
# Render
# ------------------------------------------------------------------
try:
    from IPython import get_ipython
    if get_ipython() is not None and display is not None:
        display(dot)
except (NameError, ImportError):
    pass

try:
    dot.render("architecture", view=False, cleanup=False)
    print("Saved: architecture.png")
except ExecutableNotFound:
    print("Graphviz 'dot' not found. Install the system package, then run this script again.")
    print("  macOS:  brew install graphviz")
    print("  Linux:  sudo apt install graphviz")
    raise SystemExit(1)
