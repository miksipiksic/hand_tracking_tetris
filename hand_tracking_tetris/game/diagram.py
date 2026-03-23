import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from graphviz import Digraph

dot = Digraph("TetrisIntegration", format="png")

# Globalne postavke
dot.attr(rankdir="TB", splines="ortho", bgcolor="white")
dot.attr("node", shape="rect", style="filled", fontname="DejaVu Sans", fontsize="12")

# --- Top row ---
dot.node("A", "🎥 Камера", fillcolor="#E3F2FD", color="#1565C0", fontcolor="#0D47A1")
dot.node("B", "🖐️ MediaPipe Hands", fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#1B5E20")
dot.node("C", "⚙️ StandardScaler", fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#1B5E20")

# --- Middle ---
dot.node("D", "🤖 MLP Класификатор", fillcolor="#F3E5F5", color="#6A1B9A", fontcolor="#4A148C")

# --- Bottom row ---
dot.node("E", "🔤 LabelEncoder", fillcolor="#F3E5F5", color="#6A1B9A", fontcolor="#4A148C")
dot.node("F", "🕹️ Логика игре Тетрис", fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#1B5E20")
dot.node("G", "🖥️ Визуелни приказ", fillcolor="#FFF3E0", color="#EF6C00", fontcolor="#E65100")

# --- Connections ---
dot.edge("A", "B")
dot.edge("B", "C")
dot.edge("C", "D")
dot.edge("D", "E")
dot.edge("E", "F")
dot.edge("F", "G")

# --- Force ranks (alignment by rows) ---
with dot.subgraph() as top:
    top.attr(rank="same")
    top.node("A")
    top.node("B")
    top.node("C")

with dot.subgraph() as bottom:
    bottom.attr(rank="same")
    bottom.node("E")
    bottom.node("F")
    bottom.node("G")

# --- Output ---
dot.render("tetris_integration_u_shape", cleanup=True)
print("✅ Dijagram sačuvan kao tetris_integration_u_shape.png")
