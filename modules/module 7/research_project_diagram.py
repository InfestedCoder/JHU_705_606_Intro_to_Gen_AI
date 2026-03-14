from graphviz import Digraph
from graphviz.backend.execute import ExecutableNotFound
from IPython.display import display

dot = Digraph("solver_log_copilot_pipeline", format="png")
dot.attr(rankdir="TB", splines="true", nodesep="0.5", ranksep="0.7")
dot.attr("node", shape="box", style="rounded", fontsize="14")
dot.attr("edge", arrowsize="0.8")

# Nodes
dot.node("A", "Raw Solver Logs")
dot.node("B", "Cleaning and Chunking")
dot.node("C", "Dataset Builder")

dot.node("D", "Encoder Training for\nClassification")
dot.node("E", "Seq2Seq / Decoder Training\nfor Explanation")

dot.node("F", "Predicted Diagnostic Labels")
dot.node("G", "Generated Explanation")

dot.node("H", "Final Diagnostic Report")

# Edges
dot.edge("A", "B")
dot.edge("B", "C")

dot.edge("C", "D")
dot.edge("C", "E")

dot.edge("D", "F")
dot.edge("E", "G")

dot.edge("F", "H")
dot.edge("G", "H")

# In Jupyter: show in notebook. From command line: save and open the chart.
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        display(dot)
except NameError:
    pass
try:
    dot.render("research_project_diagram", view=True, cleanup=True)
except ExecutableNotFound:
    print("Graphviz 'dot' not found. Install the system package, then run this script again.")
    print("  macOS:  brew install graphviz")
    print("  Linux: sudo apt install graphviz   (or your distro's equivalent)")
    raise SystemExit(1)