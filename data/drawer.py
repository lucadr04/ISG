import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

def get_row(ntype):
    if ntype == "Activity":
        return 1
    elif ntype == "time:timestamp":
        return 2
    elif ntype == "org:resource":
        return 3
    else:
        return 0

def drawer(g_original):
    g = g_original.clone()
    G = nx.Graph()

    node_id = {}
    node_label = {}
    node_color = {}
    edge_label = {}
    counter = 0

    type_colors = {
        ntype: plt.cm.tab10(i % 10)
        for i, ntype in enumerate(g.node_types)
    }

    import torch

    def format_features(x_row):
        t = torch.tensor(x_row) if not isinstance(x_row, torch.Tensor) else x_row
        # one-hot: un solo 1, resto 0
        if t.sum() == 1.0 and ((t == 0) | (t == 1)).all():
            return f"class={t.argmax().item()}"
        # scalare singolo
        if t.numel() == 1:
            v = t.item()
            return str(int(v)) if float(v).is_integer() else f"{v:.4f}"
        # vettore multi-feature
        parts = []
        for v in t.tolist():
            parts.append(str(int(v)) if float(v).is_integer() else f"{v:.3f}")
        return ", ".join(parts)

    for ntype in g.node_types:
        n = g[ntype].x.shape[0]
        for i in range(n):
            nid = counter
            node_id[(ntype, i)] = nid
            vals = g[ntype].x[i].tolist()
            vals_str = format_features(g[ntype].x[i])
            node_label[nid] = f"{ntype}[{i}]\n{vals_str}"
            node_color[nid] = type_colors[ntype]
            G.add_node(nid)
            counter += 1

    for (src_type, rel, dst_type) in g.edge_types:
        if rel.startswith("rev_"):
            continue
        edge_index = g[(src_type, rel, dst_type)].edge_index
        for j in range(edge_index.shape[1]):
            src = node_id[(src_type, edge_index[0, j].item())]
            dst = node_id[(dst_type, edge_index[1, j].item())]
            if src != dst:
                G.add_edge(src, dst)
                edge_label[(src, dst)] = rel

    # --- custom row layout ---
    row_counters = {0: 0, 1: 0, 2: 0, 3: 0}
    row_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for ntype in g.node_types:
        row_counts[get_row(ntype)] += g[ntype].x.shape[0]

    pos = {}
    for (ntype, i), nid in node_id.items():
        row = get_row(ntype)
        col = row_counters[row]
        total = row_counts[row]
        # center nodes horizontally within each row
        x = (col*1) + col - (total - 1) / 2.0
        y = -row * 6.0
        pos[nid] = (x, y)
        row_counters[row] += 1

    row_labels = {0: "Other attributes", 1: "Activities", 2: "Timestamps", 3: "Resources"}
    x_min = min(p[0] for p in pos.values()) - 1

    fig, ax = plt.subplots(figsize=(15, 5))
    node_color=[node_color[n] for n in G.nodes()]
    nx.draw(G, pos, labels=node_label, with_labels=True,
            node_size=2500, font_size=6, node_color=[node_color[n] for n in G.nodes()],
            edgecolors="black", linewidths=1.0, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label,
                              font_size=5, label_pos=0.3, ax=ax)

    for row, label in row_labels.items():
        ax.text(x_min, -row * 6.0, label, fontsize=9,
                va="center", ha="right", color="gray", style="italic")

    plt.tight_layout()
    plt.show()