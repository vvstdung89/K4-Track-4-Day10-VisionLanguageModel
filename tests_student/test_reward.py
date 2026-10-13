import json

import pytest

from src.reward import completion_text, gt_from_objects, iou_reward, reward


def _ans(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(b)} for lab, b in boxes])


GT = [{"label": "pothole", "bbox": (100.0, 100.0, 200.0, 200.0)},
      {"label": "crack", "bbox": (500.0, 500.0, 700.0, 600.0)}]


def test_perfect_prediction_gets_full_reward():
    assert reward(_ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))), GT) == pytest.approx(1.0)
    # order and ```json fences do not matter (referee parser)
    fenced = "```json\n" + _ans(("crack", (500, 500, 700, 600)), ("pothole", (100, 100, 200, 200))) + "\n```"
    assert reward(fenced, GT) == pytest.approx(1.0)


def test_broken_json_gets_zero():
    assert reward('[{"label": "pothole", "bbox_2d": [100, 100, 200', GT) == 0.0      # cut at max length
    assert reward("there is a pothole", GT) == 0.0
    assert reward(_ans(("pothole", (200, 100, 100, 200))), GT) == 0.0                 # x2 <= x1: invalid box


def test_wrong_class_only_gets_format_reward():
    pred = _ans(("crack", (100, 100, 200, 200)), ("pothole", (500, 500, 700, 600)))  # right places, swapped labels
    assert reward(pred, GT) == pytest.approx(0.2)


def test_extra_boxes_are_penalised_through_the_denominator():
    one_right = _ans(("pothole", (100, 100, 200, 200)))
    assert reward(one_right, GT) == pytest.approx(0.2 + 0.8 * 1 / 2)                  # one GT missed
    extra = _ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600)),
                 ("crack", (0, 0, 50, 50)), ("manhole", (800, 800, 900, 900)))
    assert reward(extra, GT) == pytest.approx(0.2 + 0.8 * 2 / 4)                      # 2 invented boxes
    # duplicates cannot both match the same GT box
    dup = _ans(("pothole", (100, 100, 200, 200)), ("pothole", (100, 100, 200, 200)))
    assert reward(dup, GT[:1]) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_empty_cases():
    assert reward("[]", []) == 1.0                                                     # nothing to find, nothing found
    assert reward("[]", GT) == pytest.approx(0.2)                                      # valid but finds nothing
    assert reward(_ans(("pothole", (1, 1, 2, 2))), []) == pytest.approx(0.2)           # invents a box on empty GT


def test_partial_iou_is_rewarded_only_above_threshold():
    shifted = _ans(("pothole", (110, 100, 210, 200)))                                  # IoU = 90/110
    assert reward(shifted, GT[:1]) == pytest.approx(0.2 + 0.8 * (90 / 110))
    far = _ans(("pothole", (160, 100, 260, 200)))                                      # IoU = 40/160 < 0.5
    assert reward(far, GT[:1]) == pytest.approx(0.2)


def test_gt_from_objects_and_trl_wrapper():
    gt = gt_from_objects([{"label": "pothole", "yolo": [0.15, 0.15, 0.1, 0.1]}])
    assert gt[0]["label"] == "pothole" and gt[0]["bbox"] == pytest.approx((100, 100, 200, 200))
    completions = [[{"role": "assistant", "content": _ans(("pothole", (100, 100, 200, 200)))}],
                   [{"role": "assistant", "content": [{"type": "text", "text": "not json"}]}]]
    gt_col = [json.dumps(gt)] * 2
    assert iou_reward(completions, gt_col) == pytest.approx([1.0, 0.0])
    assert completion_text("[]") == "[]"
