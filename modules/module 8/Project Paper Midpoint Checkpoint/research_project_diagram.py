from graphviz import Digraph
from graphviz.backend.execute import ExecutableNotFound

try:
    from IPython.display import display
except ImportError:
    display = None

dot = Digraph("solver_log_copilot_pipeline", format="png")
dot.attr(rankdir="TB", splines="true", nodesep="0.6", ranksep="0.8")
dot.attr("node", shape="box", style="rounded", fontsize="13", fontname="Helvetica")
dot.attr("edge", arrowsize="0.8")
dot.graph_attr["dpi"] = "300"

# -------------------------------------------------------------------
# Color palette
# -------------------------------------------------------------------
INPUT_ATTR   = {"fillcolor": "#d5e8d4", "style": "filled,rounded"}   # green
STAGE_ATTR   = {"fillcolor": "#dae8fc", "style": "filled,rounded"}   # blue
OUTPUT_ATTR  = {"fillcolor": "#fff2cc", "style": "filled,rounded"}   # orange/yellow
AUX_ATTR     = {"fillcolor": "#e8e8e8", "style": "filled,rounded"}   # gray

# -------------------------------------------------------------------
# Nodes  (match paper Section 4.1, four-stage pipeline)
# -------------------------------------------------------------------
dot.node("A", "Raw Solver Logs",                          **INPUT_ATTR)
dot.node("B", "Log Parser\n(regex extraction)",            **STAGE_ATTR)
dot.node("C", "Structured Representation\n(JSON event stream)", **STAGE_ATTR)
dot.node("D", "Transformer Encoder\n(embedding + retrieval)",  **STAGE_ATTR)

dot.node("RAG", "Solver Behavior\nKnowledge Base (RAG)",  **AUX_ATTR)

dot.node("E", "Bottleneck Classifier\n(multi-label)",     **STAGE_ATTR)
dot.node("F", "Diagnostic Generator\n(seq2seq / LLM)",    **STAGE_ATTR)

dot.node("G", "Predicted\nDiagnostic Labels",             **OUTPUT_ATTR)
dot.node("H", "Generated\nExplanation",                   **OUTPUT_ATTR)

dot.node("I", "Diagnostic Report",                        **OUTPUT_ATTR)

# -------------------------------------------------------------------
# Edges
# -------------------------------------------------------------------
# main pipeline
dot.edge("A", "B")
dot.edge("B", "C")
dot.edge("C", "D")

# RAG feed
dot.edge("RAG", "D", style="dashed")

# split into two heads
dot.edge("D", "E")
dot.edge("D", "F")

# outputs
dot.edge("E", "G")
dot.edge("F", "H")

# merge
dot.edge("G", "I")
dot.edge("H", "I")

# -------------------------------------------------------------------
# Stage labels (invisible helper nodes on the left)
# -------------------------------------------------------------------
dot.node("L0", "Input",   shape="plaintext", fontsize="10", fontcolor="gray", fontname="Helvetica-Oblique")
dot.node("L1", "Stage 1", shape="plaintext", fontsize="10", fontcolor="gray", fontname="Helvetica-Oblique")
dot.node("L2", "Stage 2", shape="plaintext", fontsize="10", fontcolor="gray", fontname="Helvetica-Oblique")
dot.node("L3", "Stage 3", shape="plaintext", fontsize="10", fontcolor="gray", fontname="Helvetica-Oblique")
dot.node("L4", "Stage 4", shape="plaintext", fontsize="10", fontcolor="gray", fontname="Helvetica-Oblique")

# align labels to their stages
dot.edge("L0", "L1", style="invis")
dot.edge("L1", "L2", style="invis")
dot.edge("L2", "L3", style="invis")
dot.edge("L3", "L4", style="invis")

# Use subgraphs with rank=same to align labels horizontally with stages
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("L0")
    s.node("A")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("L1")
    s.node("B")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("L2")
    s.node("C")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("L3")
    s.node("D")
    s.node("RAG")

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node("L4")
    s.node("E")
    s.node("F")

# -------------------------------------------------------------------
# Render
# -------------------------------------------------------------------
# In Jupyter: show in notebook. From command line: save the file.
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
