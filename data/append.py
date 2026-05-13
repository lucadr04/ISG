


def to_onehot(logits, n_classes):
    idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
    oh = torch.zeros(1, n_classes)
    oh[0, idx] = 1.0
    return oh

def add_edge(g, rel, src, dst):
  new_e = torch.tensor([[src], [dst]])
  if hasattr(g[rel], 'edge_index') and g[rel].edge_index is not None:
      g[rel].edge_index = torch.cat([g[rel].edge_index, new_e], dim=1)
  else:
      g[rel].edge_index = new_e

def find_node_index(g, ntype, predicted_tensor):
    """Trova l'indice del nodo di tipo ntype che ha lo stesso argmax del tensore predetto."""
    argmax_pred = predicted_tensor.argmax().item()
    for i in range(g[ntype].x.shape[0]):
        if g[ntype].x[i].argmax().item() == argmax_pred:
            return i
    return None  # non trovato — non dovrebbe succedere

def add_end_activity(g):
    """
    Appends a synthetic END Activity node to an existing graph g,
    wiring it into the existing edge structure.
    Meant to be called on the full/completed graph.
    """
    # --- Build END one-hot ---
    end_feat = torch.tensor(
        ONE_HOT_ENCODERS["Activity"].transform([["END"]]).toarray(),
        dtype=torch.float32
    )  # shape [1, n_activity_classes]

    # --- Snapshot current last activity index, then append ---
    old_last_act_idx = g["Activity"].x.shape[0] - 1
    g["Activity"].x = torch.cat([g["Activity"].x, end_feat], dim=0)
    new_act_idx = old_last_act_idx + 1

    # --- Wire up edges ---
    for (src, rel, dst) in g.edge_types:

        # Activity chain: last real event → END
        if src == "Activity" and dst == "Activity":
            add_edge(g, (src, rel, dst), old_last_act_idx, new_act_idx)

        # Activity → other node type
        elif src == "Activity" and dst != "Activity":
            """ I am most probaably doing something useless here """
            # static node (single node, index 0) vs dynamic (match by position)
            target_idx = 0 if g[dst].x.shape[0] == 1 else old_last_act_idx
            add_edge(g, (src, rel, dst), new_act_idx, target_idx)

    return g

def add_end_activity_ground(t): 
    end_feat = torch.tensor(
        ONE_HOT_ENCODERS["Activity"].transform([["END"]]).toarray(),
        dtype=torch.float32
    )
    t["Activity"] = torch.cat([t["Activity"], end_feat], dim=0)

def append_predicted_event(g, out, outputcat, outputreal):

  old_idx = {}   # indici dei nodi a cui appendere l'output
  new_idx = {}   # indici dei nodi da appendere

  old_idx = {ntype: g[ntype].x.shape[0] - 1 for ntype in g.node_types}  # snapshot iniziale

  print_edges(g)

  for k in outputcat:
      new_feat = to_onehot(out[k], outputcat[k])

      if k in resource_types:
          existing = find_node_index(g, k, new_feat)
          if existing is not None:
              new_idx[k] = existing  # riusato
          else:
              g[k].x = torch.cat([g[k].x, new_feat], dim=0)
              new_idx[k] = old_idx[k] + 1 # appena creato
      else:
          g[k].x = torch.cat([g[k].x, new_feat], dim=0)
          new_idx[k] = old_idx[k] + 1

  for k in outputreal:
      new_feat = out[k].reshape(1, 1)

      if k in resource_types:
          existing = find_node_index(g, k, new_feat)
          if existing is not None:
              new_idx[k] = existing  # riusato
          else:
              g[k].x = torch.cat([g[k].x, new_feat], dim=0)
              new_idx[k] = old_idx[k] + 1  # appena creato
      else:
          g[k].x = torch.cat([g[k].x, new_feat], dim=0)
          new_idx[k] = old_idx[k] + 1

  # --- ricava gli archi da aggiungere dalla struttura esistente ---
  for (src, rel, dst) in g.edge_types:

      # Concateno predizioni
      if src == dst and src in predicted_types:
          if new_idx[src] == old_idx[src] + 1:
              add_edge(g, (src, rel, dst), old_idx[src], new_idx[src])

      # Activity → attributo predetto
      elif src == "Activity" and dst in predicted_types:
          add_edge(g, (src, rel, dst), new_idx[src], new_idx[dst])

      # Activity → costante
      elif src == "Activity" and dst not in predicted_types:
          add_edge(g, (src, rel, dst), new_idx[src], 0)

  print_edges(g)


  return g