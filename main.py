import argparse
import platform
import json
import logging
import math
import os
import random
import shutil
import sys
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch_geometric
import torch.nn as nn
from rapidfuzz.distance import DamerauLevenshtein
from torch.nn import Linear, Module, ModuleDict, ModuleList
from torch.nn.functional import l1_loss
from torch_geometric.data import Batch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, HeteroConv, global_mean_pool
from torch_geometric.transforms import ToUndirected
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

pd.set_option("display.max_columns", None)


# --------------------------------------------------------------------------- #
# Fixed config
# --------------------------------------------------------------------------- #
PATIENCE = 25
NUM_EPOCHS = 200
TOT_TRIALS = 25            
NUM_TEST_RUNS = 10        
MAX_INFERENCE_STEPS = 100
SEED = 0


# --------------------------------------------------------------------------- #
# CLI args
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test the HGNN on a process-mining event log dataset."
    )
    parser.add_argument("dataset", help="Dataset name, e.g. bpi_2012")
    parser.add_argument(
        "hyperopt",
        nargs="?",
        default="false",
        help="true/false - run Ax hyperparameter optimization first (default: false)",
    )
    args = parser.parse_args()
    do_hyperopt = args.hyperopt.strip().lower() in ("true", "1", "yes", "y", "t")
    return args.dataset, do_hyperopt

DATASET, DO_HYPEROPT = parse_args()
print(f"Begin training on {DATASET} (ax: {DO_HYPEROPT})")
print(f"to follow use 'tail -f results/{DATASET}/run.log'")


# --------------------------------------------------------------------------- #
# Pathings
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.environ.get("ISG_ROOT_PATH", f"{os.path.dirname(SCRIPT_DIR)}/ISG")

PROCESSED_PATH = f"{ROOT_PATH}/data/datasets/processed/{DATASET}"
GRAPHS_PATH = f"{ROOT_PATH}/data/datasets/graphs/{DATASET}"
MODEL_PATH = f"{ROOT_PATH}/data/datasets/models/{DATASET}"
RESULT_PATH = f"{ROOT_PATH}/results/{DATASET}"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logger = logging.getLogger("hgnn_logger")
logger.setLevel(logging.INFO)
logger.propagate = False 

file_handler = logging.FileHandler(f"{RESULT_PATH}/run.log", mode="a")
console_handler = logging.StreamHandler(sys.stdout)

# formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Dataset: {DATASET} | hyperopt: {DO_HYPEROPT} | device: {device}")

# --------------------------------------------------------------------------- #
# Runtime Specifics
# --------------------------------------------------------------------------- #
def log_run_specifics(logger, device, dataset):
    logger.info("="*50)
    logger.info("RUNTIME SPECIFICS")
    logger.info("="*50)
    
    # 1. System & Python
    logger.info(f"OS           : {platform.system()} {platform.release()}")
    logger.info(f"Python       : {platform.python_version()}")
    
    # 2. Deep Learning Libraries
    logger.info(f"PyTorch      : {torch.__version__}")
    logger.info(f"PyG          : {torch_geometric.__version__}")
    
    # 3. Hardware details
    logger.info(f"Device       : {device}")
    if device.type == 'cuda':
        logger.info(f"GPU          : {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA Version : {torch.version.cuda}")
    
    # 4. Script Configurations
    logger.info(f"Dataset      : {dataset}")
    logger.info(f"Hyperopt     : {DO_HYPEROPT}")
    logger.info(f"Epochs       : {NUM_EPOCHS} | Patience: {PATIENCE}")
    logger.info(f"Trials       : {TOT_TRIALS} | Test Runs: {NUM_TEST_RUNS}")
    logger.info(f"Seed         : {SEED}")
    logger.info("="*50)

# Execute the logging block
log_run_specifics(logger, device, DATASET)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.serialization.add_safe_globals([HeteroData])


# --------------------------------------------------------------------------- #
# Dataset and outputs (for now act, res, time, but can be edit)
# --------------------------------------------------------------------------- #
with open(f"{ROOT_PATH}/data/dataset_features.json", "r") as f:
    dataset_info = json.load(f)[DATASET]

categorical_columns = dataset_info["categorical"]
real_value_columns = dataset_info["numerical"]

tab_all = pd.read_csv(f"{PROCESSED_PATH}/{DATASET}_processed_all.csv")
list_unique = {k: list(tab_all[k].unique()) for k in categorical_columns}

outputcat = {k: len(list_unique[k]) for k in categorical_columns if k in ("Activity", "org:resource")}
outputreal = [k for k in real_value_columns if k == "time:timestamp"]
outputcat["Activity"] = outputcat["Activity"] + 1  # +1 for the END node

# --------------------------------------------------------------------------- #
# Graphs
# --------------------------------------------------------------------------- #
logger.info("Loading graphs...")
X_TRAIN = torch.load(f"{GRAPHS_PATH}/train_set.pt", weights_only=False)
X_VALID = torch.load(f"{GRAPHS_PATH}/validation_set.pt", weights_only=False)
X_TEST = torch.load(f"{GRAPHS_PATH}/test_set.pt", weights_only=False)

_node_types, _edge_types = set(), set()
for _g in X_TRAIN:
    _n, _e = _g.metadata()
    _node_types.update(_n)
    _edge_types.update(_e)
for _g in X_VALID:
    _n, _e = _g.metadata()
    _node_types.update(_n)
    _edge_types.update(_e)
for _g, _ in X_TEST:
    _n, _e = _g.metadata()
    _node_types.update(_n)
    _edge_types.update(_e)

NODE_TYPES = list(_node_types)
EDGE_TYPES = list(_edge_types)
logger.info(f"{len(X_TRAIN)} train / {len(X_VALID)} valid / {len(X_TEST)} test traces loaded")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class HGNN(Module):
    def __init__(self, output_cat, output_real, nodes_relations, parameters):
        super().__init__()

        hid = parameters["hid"]
        layers = parameters["layers"]
        aggregation = parameters["aggregation"]

        self.output_cat = output_cat
        self.output_real = output_real

        self.convs = ModuleList()
        for _ in range(layers):
            conv = HeteroConv(
                {
                    relation: GATv2Conv((-1, -1), hid, 1, False, add_self_loops=False, residual=False)
                    for relation in nodes_relations
                },
                aggr=aggregation,
            )
            self.convs.append(conv)

        self.FC = ModuleDict()
        for k in output_cat:
            self.FC[k] = Linear(hid, output_cat[k])  # logits
        for k in output_real:
            self.FC[k] = Linear(hid, 1)  # scalar

    def forward(self, batch):
        x_dict = batch.x_dict
        for conv in self.convs:
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        output = {}
        for k in self.output_cat:
            output[k] = global_mean_pool(x_dict[k], batch[k].batch)
            output[k] = self.FC[k](output[k])
        for k in self.output_real:
            output[k] = global_mean_pool(x_dict[k], batch[k].batch)
            output[k] = self.FC[k](output[k]).reshape(1, -1)[0]
        return output


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _fmt_epoch_log(epoch, val_loss, accuracy, macro_f1s, MAE, pat_count, best_f1):
    acc_s = {k: round(v.item(), 3) for k, v in accuracy.items()}
    f1_s = {k: round(v.item(), 3) for k, v in macro_f1s.items()}
    mae_s = {k: round(v, 3) for k, v in MAE.items()}
    return (
        f"  epoch {epoch + 1}/{NUM_EPOCHS} | val_loss={val_loss:.4f} | "
        f"acc={acc_s} | f1={f1_s} | mae={mae_s} | "
        f"patience={pat_count}/{PATIENCE} | best_activity_f1={best_f1:.3f}"
    )


def train_hgnn(config, output_cat, output_real):
    logger.info(f"Training with config: {config}")

    net = HGNN(parameters=config, output_cat=output_cat, output_real=output_real, nodes_relations=EDGE_TYPES)
    net = net.to(device)

    losses = {k: nn.CrossEntropyLoss() for k in output_cat}
    losses.update({k: nn.L1Loss() for k in output_real})

    train_loader = DataLoader(X_TRAIN, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(X_VALID, batch_size=config["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=int(PATIENCE / 2))

    best_model, best_loss, best_f1 = None, 0, 0
    pat_count = 0
    torch.cuda.empty_cache()

    for epoch in range(NUM_EPOCHS):
        net.train()
        for x in train_loader:
            x = x.to(device)
            labels = x.y
            optimizer.zero_grad()
            outputs = net(x)
            losses_step = {k: losses[k](outputs[k], labels[k]) for k in losses}
            total_loss = sum(losses_step.values())
            total_loss.backward()
            optimizer.step()

        # -- Validation --
        running_total_loss = []
        predictions_categorical = {k: [] for k in output_cat}
        target_categorical = {k: [] for k in output_cat}
        prediction_numerical = {k: [] for k in output_real}
        target_numerical = {k: [] for k in output_real}

        net.eval()
        with torch.no_grad():
            for x in valid_loader:
                x = x.to(device)
                labels = x.y
                outputs = net(x)
                losses_step = {k: losses[k](outputs[k], labels[k]) for k in losses}
                running_total_loss.append(sum(losses_step.values()))

                for k in output_cat:
                    predictions_categorical[k].append(torch.argmax(torch.softmax(outputs[k], dim=1), 1))
                    target_categorical[k].append(labels[k])
                for k in output_real:
                    prediction_numerical[k].append(outputs[k])
                    target_numerical[k].append(labels[k])

        val_loss = sum(running_total_loss) / len(running_total_loss)
        lr_scheduler.step(val_loss)

        for k in predictions_categorical:
            predictions_categorical[k] = torch.cat(predictions_categorical[k])
            target_categorical[k] = torch.cat(target_categorical[k])
        for k in prediction_numerical:
            prediction_numerical[k] = torch.cat(prediction_numerical[k])
            target_numerical[k] = torch.cat(target_numerical[k])

        macro_f1s = {
            k: multiclass_f1_score(
                predictions_categorical[k], target_categorical[k].squeeze(),
                num_classes=output_cat[k], average="macro",
            )
            for k in output_cat
        }
        f1_activity = macro_f1s["Activity"]
        accuracy = {
            k: multiclass_accuracy(predictions_categorical[k], target_categorical[k].squeeze(), num_classes=output_cat[k])
            for k in output_cat
        }
        MAE = {k: l1_loss(prediction_numerical[k], target_numerical[k]).item() for k in output_real}

        if epoch == 0:
            best_model, best_loss, best_f1 = deepcopy(net), val_loss, f1_activity
            logger.info(_fmt_epoch_log(epoch, val_loss, accuracy, macro_f1s, MAE, pat_count, best_f1))
        else:
            if val_loss < best_loss:
                best_loss, best_model, pat_count = val_loss, deepcopy(net), -1
            if best_f1 < f1_activity:
                best_f1 = f1_activity
            logger.info(_fmt_epoch_log(epoch, val_loss, accuracy, macro_f1s, MAE, pat_count, best_f1))
            if pat_count == PATIENCE:
                return {"valid_loss": best_loss.item()}, best_model
        pat_count += 1

    logger.info(_fmt_epoch_log(NUM_EPOCHS - 1, val_loss, accuracy, macro_f1s, MAE, pat_count, best_f1))
    return {"valid_loss": best_loss.item()}, best_model


# --------------------------------------------------------------------------- #
# Testing helpers
# --------------------------------------------------------------------------- #
def get_one_hot_encoder(dataset, key):
    values = dataset[key].unique()
    if key == "Activity":
        values = np.union1d(values, ["END"])
    values = values.astype(np.str_).reshape([len(values), 1])
    encoder = sklearn.preprocessing.OneHotEncoder()
    encoder.fit(values)
    return encoder


def to_onehot(logits, n_classes):
    idx = torch.argmax(logits, dim=1).item()
    oh = torch.zeros(1, n_classes)
    oh[0, idx] = 1.0
    return oh


ONE_HOT_ENCODERS = {k: get_one_hot_encoder(tab_all, k) for k in categorical_columns}
END_IDX = int(ONE_HOT_ENCODERS["Activity"].transform([["END"]]).argmax())
logger.info(f"Activity categories: {ONE_HOT_ENCODERS['Activity'].categories_[0].tolist()}")
logger.info(f"END class index: {END_IDX}")

CONNECTED_TYPES = set(outputcat.keys()) | set(outputreal) | {"Activity"}


def compute_edges_indexs(node_features, prefix_len):
    res = {}
    keys = node_features.keys()
    for k in keys:
        n_k = len(node_features[k])
        if n_k != 1:
            if k == "Activity":
                res[(k, "followed_by", k)] = [[i, i + 1] for i in range(prefix_len - 1)]
                for k2 in keys:
                    if k2 != k:
                        n_k2 = len(node_features[k2])
                        if n_k2 == 1:
                            res[(k, "related_to", k2)] = [[i, 0] for i in range(prefix_len)]
                        else:
                            last = n_k2 - 1
                            res[(k, "related_to", k2)] = [[i, min(i, last)] for i in range(prefix_len)]
            else:
                res[(k, "related_to", k)] = [[i, i + 1] for i in range(n_k - 1)]
    return res


def append_predicted_event(g, out, output_cat, output_real):
    new_g = HeteroData()

    for node_type in g.node_types:
        existing = g[node_type].x.detach().cpu()

        if node_type == "Activity":
            new_feat = to_onehot(out[node_type], output_cat[node_type])
            new_g[node_type].x = torch.cat([existing, new_feat], dim=0)
        elif existing.shape[0] == 1:
            new_g[node_type].x = existing
        elif node_type in output_cat:
            new_feat = to_onehot(out[node_type], output_cat[node_type])
            new_g[node_type].x = torch.cat([existing, new_feat], dim=0)
        elif node_type in output_real:
            new_feat = out[node_type].reshape(1, 1).detach().cpu()
            new_g[node_type].x = torch.cat([existing, new_feat], dim=0)
        else:
            new_g[node_type].x = existing

    prefix_len = new_g["Activity"].x.shape[0]
    node_features_filtered = {
        nt: new_g[nt].x
        for nt in new_g.node_types
        if nt in CONNECTED_TYPES or new_g[nt].x.shape[0] == 1
    }
    edges = compute_edges_indexs(node_features_filtered, prefix_len)

    for rel, edge_list in edges.items():
        src_nodes = [pair[0] for pair in edge_list]
        dst_nodes = [pair[1] for pair in edge_list]
        new_g[rel].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    return ToUndirected()(new_g)


def test_hgnn(net, output_cat, output_real, max_steps=MAX_INFERENCE_STEPS):
    """
    For each trace in the test set:
      - start from the 2-event prefix
      - predict the next event
      - append it to the graph
      - repeat until END is predicted or max_steps is reached
      - compare against ground truth
    """
    net.eval()
    all_predicted_act, all_ground_truth_act = [], []
    dl_normsim_act, dl_normsim_res = [], []
    mae_remaining, trace_len, seq_acc, len_err = [], [], [], []
    end_not_reached = 0

    n = len(X_TEST)
    log_every = max(1, n // 20)

    with torch.no_grad():
        for idx, (graph, ground_truth) in enumerate(X_TEST):
            current_graph = graph.clone()
            predicted_activities, predicted_resources, predicted_timestamps = [], [], []

            for _ in range(max_steps):
                batched = Batch.from_data_list([current_graph]).to(device)
                output = net(batched)

                # Repetition penalty
                act_logits = output["Activity"].clone().squeeze()
                for act_idx in set(predicted_activities):
                    count = predicted_activities.count(act_idx)
                    act_logits[act_idx] -= math.exp(count)

                pred_act = torch.argmax(act_logits, dim=-1).item()
                pred_res = torch.argmax(output["org:resource"], dim=-1).item()
                pred_time = output["time:timestamp"].item()

                predicted_activities.append(pred_act)
                predicted_resources.append(pred_res)
                predicted_timestamps.append(pred_time)

                if pred_act == END_IDX:
                    break

                output_cpu = {k: v.cpu() for k, v in output.items()}
                current_graph = append_predicted_event(current_graph, output_cpu, output_cat, output_real)
            else:
                end_not_reached += 1

            gt_activities = [torch.argmax(v).item() for v in ground_truth["Activity"]][2:]
            gt_resources = (
                [torch.argmax(ground_truth["org:resource"][i]).item() for i in range(2, len(ground_truth["org:resource"]))]
                if "org:resource" in ground_truth and len(ground_truth["org:resource"]) > 1
                else [torch.argmax(ground_truth["org:resource"][0]).item()] * len(gt_activities)
                if "org:resource" in ground_truth
                else []
            )
            gt_timestamps = (
                [ground_truth["time:timestamp"][i].item() for i in range(2, len(ground_truth["time:timestamp"]))]
                if "time:timestamp" in ground_truth
                else []
            )

            dl_normsim_act.append(DamerauLevenshtein.normalized_similarity(predicted_activities, gt_activities))
            if gt_resources:
                dl_normsim_res.append(DamerauLevenshtein.normalized_similarity(predicted_resources, gt_resources))
            if gt_timestamps:
                mae_remaining.append(abs(predicted_timestamps[-1] - gt_timestamps[-1]))

            trace_len.append(len(gt_activities))
            seq_acc.append(int(predicted_activities == gt_activities))
            len_err.append(abs(len(predicted_activities) - len(gt_activities)))
            all_predicted_act.append(predicted_activities)
            all_ground_truth_act.append(gt_activities)

            if (idx + 1) % log_every == 0 or (idx + 1) == n:
                logger.info(f"  [test] trace {idx + 1}/{n}")

    res = {
        "n_traces": n,
        "avg_trace_len": sum(trace_len) / n,
        "max_trace_len": max(trace_len),
        "end_not_reached": end_not_reached,
        "seq_acc_mean": sum(seq_acc) / n,
        "len_err_mean": sum(len_err) / n,
        "len_err_std": float(torch.tensor(len_err, dtype=torch.float).std()),
        "dl_normsim_act_mean": sum(dl_normsim_act) / n,
        "dl_normsim_act_std": float(torch.tensor(dl_normsim_act, dtype=torch.float).std()),
        "dl_normsim_res_mean": sum(dl_normsim_res) / len(dl_normsim_res) if dl_normsim_res else None,
        "dl_normsim_res_std": float(torch.tensor(dl_normsim_res, dtype=torch.float).std()) if dl_normsim_res else None,
        "mae_remaining_mean": sum(mae_remaining) / len(mae_remaining) if mae_remaining else None,
        "mae_remaining_std": float(torch.tensor(mae_remaining, dtype=torch.float).std()) if mae_remaining else None,
    }

    report_str = (
        f"\n{'=' * 55}\n"
        f"  AUTOREGRESSIVE INFERENCE RESULTS ({n} traces)\n"
        f"{'=' * 55}\n"
        f"  - End Not Reached               : {res['end_not_reached']} times, ({(res['end_not_reached'] / n) * 100:.1f}%)\n"
        f"  - Avg & Max Trace Length (GT)   : {res['avg_trace_len']:.2f}, {res['max_trace_len']}\n"
        f"  - Sequence Accuracy             : {res['seq_acc_mean'] * 100:.2f}% perfect matches\n"
        f"  - Length Error (MAE)            : {res['len_err_mean']:.2f} +/- {res['len_err_std']:.2f} events\n"
        f"{'-' * 55}\n"
        f"  - DL Similarity Activity        : {res['dl_normsim_act_mean'] * 100:.2f}% +/- {res['dl_normsim_act_std'] * 100:.2f}%\n"
    )
    if res["dl_normsim_res_mean"] is not None:
        report_str += f"  - DL Similarity Resource        : {res['dl_normsim_res_mean'] * 100:.2f}% +/- {res['dl_normsim_res_std'] * 100:.2f}%\n"
    if res["mae_remaining_mean"] is not None:
        report_str += f"  - MAE Remaining Time            : {res['mae_remaining_mean']:.4f} +/- {res['mae_remaining_std']:.4f} (time units)\n"
    report_str += f"{'=' * 55}\n"

    logger.info(report_str)
    return res, all_predicted_act, all_ground_truth_act, report_str


def create_df(results):
    data = {k: [r[k] for r in results] for k in results[0]}
    df = pd.DataFrame(data=data)
    return df, df.mean(), df.std()


# --------------------------------------------------------------------------- #
# Test(count NUM_TEST_RUNS)
# --------------------------------------------------------------------------- #
def test_multi(config, output_cat, output_real, num_runs=NUM_TEST_RUNS):
    all_metrics = []
    run_info = []
    tmp_dir = f"{MODEL_PATH}/_tmp_test_runs"
    os.makedirs(tmp_dir, exist_ok=True)

    for i in range(num_runs):
        logger.info(f"=== Run {i + 1}/{num_runs} ===")
        _, net = train_hgnn(config, output_cat, output_real)
        res, pred_act, gt_act, report_str = test_hgnn(net, output_cat, output_real)

        state_path = f"{tmp_dir}/run_{i}.pt"
        torch.save(net.state_dict(), state_path)
        del net
        torch.cuda.empty_cache()

        all_metrics.append(res)
        run_info.append(
            {"res": res, "pred_act": pred_act, "gt_act": gt_act, "report_str": report_str, "state_path": state_path}
        )
        logger.info(f"Run {i + 1} metrics: {res}")

    # -- All runs' metrics: always saved in full --
    results_table, means, stds = create_df(all_metrics)
    results_table.to_csv(f"{RESULT_PATH}/results.csv", sep=",", index=False)
    logger.info(f"Saved metrics for all {num_runs} runs to results.csv")

    # -- Best run (activity prediction) --
    best_idx = max(range(num_runs), key=lambda i: run_info[i]["res"]["dl_normsim_act_mean"])
    best = run_info[best_idx]
    logger.info(
        f"Best model: run {best_idx + 1}/{num_runs} "
        f"(dl_normsim_act_mean={best['res']['dl_normsim_act_mean']:.4f}, "
        f"seq_acc_mean={best['res']['seq_acc_mean']:.4f})"
    )

    with open(f"{RESULT_PATH}/inference_report.txt", "w") as f:
        f.write(best["report_str"])
    with open(f"{RESULT_PATH}/all_predicted_act.json", "w") as f:
        json.dump(best["pred_act"], f)
    with open(f"{RESULT_PATH}/all_ground_truth_act.json", "w") as f:
        json.dump(best["gt_act"], f)
    shutil.copy(best["state_path"], f"{MODEL_PATH}/model.pt")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Saved best model's report/predictions/weights under results/{DATASET} and models/{DATASET}/model.pt")

    return all_metrics, best_idx


# --------------------------------------------------------------------------- #
# Hyperparameter optimization (Ax)
# --------------------------------------------------------------------------- #
def run_hyperopt(output_cat, output_real):
    from ax.service.managed_loop import optimize
    from ax.service.utils.report_utils import exp_to_df

    def train_evaluate(config):
        res, _ = train_hgnn(config, output_cat=output_cat, output_real=output_real)
        return res

    logger.info(f"Starting Ax hyperparameter search ({TOT_TRIALS} trials)...")
    best_parameters, values, experiment, _ = optimize(
        parameters=[
            {"name": "hid", "type": "choice", "values": [128, 256, 512], "value_type": "int", "is_ordered": True, "sort_values": False},
            {"name": "batch_size", "type": "choice", "values": [256, 512, 1024], "value_type": "int", "is_ordered": True, "sort_values": False},
            {"name": "layers", "type": "choice", "values": [2, 4], "value_type": "int", "is_ordered": True, "sort_values": False},
            {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "value_type": "float", "log_scale": True},
            {"name": "aggregation", "type": "choice", "values": ["sum", "mean", "max"], "value_type": "str"},
        ],
        evaluation_function=train_evaluate,
        objective_name="valid_loss",
        arms_per_trial=1,
        minimize=True,
        random_seed=123,
        total_trials=TOT_TRIALS,
    )

    means, _covariances = values
    logger.info(f"Best hyperparameters found: {best_parameters}")
    logger.info(f"Predicted valid_loss: {means}")

    results = exp_to_df(experiment).sort_values(by="valid_loss")
    results.to_csv(f"{RESULT_PATH}/hyp_params_search.csv", sep=",", index=False)

    # NOTE: kept as "best_parameters.csv" for consistency with previous runs,
    # even though the content is JSON, not CSV.
    with open(f"{RESULT_PATH}/best_parameters.csv", "w") as f:
        json.dump(best_parameters, f, indent=4)

    return best_parameters


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    best_params_path = f"{RESULT_PATH}/best_parameters.csv"

    if DO_HYPEROPT:
        best_parameters = run_hyperopt(outputcat, outputreal)
    else:
        if not os.path.exists(best_params_path):
            logger.error(
                f"No saved hyperparameters found at {best_params_path}. "
                f"Run once with hyperopt enabled first, e.g.:\n"
                f"    python {os.path.basename(__file__)} {DATASET} true"
            )
            sys.exit(1)
        with open(best_params_path, "r") as f:
            best_parameters = json.load(f)
        logger.info(f"Reusing saved hyperparameters: {best_parameters}")

    test_multi(best_parameters, outputcat, outputreal, num_runs=NUM_TEST_RUNS)
    logger.info("Done.")


if __name__ == "__main__":
    main()