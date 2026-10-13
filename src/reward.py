"""IoU reward for E5 (GRPO), GUIDE step 9. Pure Python: testable without a GPU.

reward = 0                                     if the output is not valid JSON (referee's parser)
         1                                     if there is no GT box and no predicted box
         0.2 + 0.8 * sum_IoU / max(n_gt, n_pred, 1)   otherwise
sum_IoU sums the IoU of greedy same-class pairs (highest IoU first, IoU >= 0.5 only), the same matching
as scripts/score.py. Dividing by max(n_gt, n_pred) penalises both missed and invented boxes.
"""
import json

from scripts.score import iou_xyxy, parse_output

IOU_THR = 0.5


def gt_from_objects(objects):
    """dataset objects (YOLO xc, yc, w, h in [0, 1]) -> [{"label", "bbox"}] on the 0-1000 scale"""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def matched_ious(gt, pred):
    """IoUs of the greedy one-to-one same-class pairs with IoU >= 0.5"""
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt) for j, p in enumerate(pred) if g["label"] == p["label"]),
                   reverse=True)
    used_g, used_p, ious = set(), set(), []
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        ious.append(v)
    return ious


def reward(completion, gt_boxes):
    """completion: raw model text. gt_boxes: [{"label", "bbox": (x1, y1, x2, y2)}] on the 0-1000 scale."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    return 0.2 + 0.8 * sum(matched_ious(gt_boxes, pred)) / max(len(gt_boxes), len(pred), 1)


def completion_text(completion):
    """trl passes conversational completions as [{"role": "assistant", "content": str | [parts]}]"""
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def iou_reward(completions, gt, **kwargs):
    """trl GRPOTrainer reward function: `gt` is the dataset column holding json.dumps(gt_boxes)."""
    return [reward(completion_text(c), [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in json.loads(g)])
            for c, g in zip(completions, gt)]
