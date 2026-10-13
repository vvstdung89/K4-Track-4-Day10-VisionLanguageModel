# Hướng dẫn hoàn thành Lab Day 10 (checklist thực hành)

> File này là **lộ trình làm bài từ đầu đến lúc nộp**: làm gì, theo thứ tự nào, sinh ra file gì, commit lúc nào, và kiểm tra ra sao.
> Kỹ thuật chi tiết (pseudo code, câu hỏi từng bước) nằm trong [`GUIDE.md`](GUIDE.md); thang điểm trong [`RUBRIC.md`](RUBRIC.md). Khi hai file mâu thuẫn, **GUIDE/RUBRIC là chuẩn**.

---

## 0. Ba quy tắc không được phạm

1. **Không sửa `scripts/` và `tests/`** (−10 điểm; giảng viên sẽ ghi đè bằng bản gốc). Mã của bạn để trong `lab_day10.ipynb` hoặc file riêng, ví dụ `src/`.
2. **Không commit** `data/`, `work/`, `.env`, adapter, trọng số (`*.safetensors`, `*.bin`, `*.pt`, `*.ckpt`). `.gitignore` đã chặn sẵn — đừng `git add -f`.
3. **Chỉ chấm `test` một lần cho mỗi setup**, và chỉ file chấm `test` mới được ghi vào `report/evidence/`. Kết quả gỡ lỗi trên `val` ghi vào `work/`.

---

## 1. Kết quả cuối cùng phải có

| File | Sinh bởi | Yêu cầu |
|---|---|---|
| `report/evidence/e0.json` | `score.py` | `split=test`, `missing_predictions=0` |
| `report/evidence/e1.json`, `e2.json`, `e3.json` | `score.py --meta …` | như trên + `meta` có `trainable_params`, `peak_vram_gb`, `train_minutes` > 0 |
| `report/evidence/e4_<biến>_<giá trị>.json` (×2) | `score.py` | đổi **đúng một** biến, chạy **cả hai** giá trị |
| `report/figures/*.png` (≥ 6) | mã của bạn | ảnh dự đoán đúng/sai, vẽ box dự đoán **và** box đúng, có chú thích |
| `REPORT.md` | bạn | bảng tổng hợp + 10 câu, **không còn chuỗi `<điền`** |
| `lab_day10.ipynb` (và/hoặc `src/*.py`) | bạn | mã thật đã chạy |
| ≥ 5 commit | git | commit theo từng bước (xem dòng "Commit" cuối mỗi mục) |

Ngưỡng điểm tự động đáng nhớ: `parse_valid_rate` của E1 **≥ 0,95**, F1(E1) − F1(E0) **≥ 0,10**.

---

## 2. Thiết lập repo và Colab (≈ 20 phút)

### 2.1 Repo GitHub công khai của bạn

```bash
# trên máy hoặc trong Colab
git clone <repo đề> lab-day10 && cd lab-day10
git remote set-url origin https://github.com/<bạn>/<tên-repo>.git   # repo PUBLIC bạn vừa tạo
git push -u origin main
```

### 2.2 Colab

1. Runtime → *Change runtime type* → **T4 GPU**. T4 **không hỗ trợ bf16** → mọi thứ dùng **fp16**.
2. Tạo Colab Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY` (tuỳ chọn `HF_TOKEN`, `GITHUB_TOKEN` để push). Bật "Notebook access".
3. Mount Drive và clone repo vào `/content`:

```python
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/lab10'          # nơi lưu adapter, projector, predictions
os.makedirs(DRIVE, exist_ok=True)
!git clone https://github.com/<bạn>/<tên-repo>.git /content/lab10
%cd /content/lab10
```

4. Cài đặt và kiểm tra:

```bash
!pip -q install -r requirements-colab.txt
!python -m pytest -q
```

> Nếu báo thiếu torchvision: cài `torchvision` khớp với bản torch có sẵn của Colab, **không** nâng torch.
> Lưu ý: sau khi bạn điền xong `REPORT.md`, test `test_report_template_has_student_markers_and_ten_questions` sẽ **fail** (nó kiểm tra mẫu còn `<điền`). Đó là bình thường — **đừng sửa test**; lúc nộp chỉ cần `check_submission.py` sạch.

### 2.3 Chống mất session

Colab free có thể ngắt bất cứ lúc nào. Sau **mỗi** lần train/suy luận, chép ngay lên Drive:

```python
!cp -r work/e1_adapter work/e1_extra_trainable.pt work/e1.jsonl work/e1_run.json $DRIVE/ 2>/dev/null
```

Khi mở session mới: clone lại repo, chạy lại `prepare_data.py` (split **cố định**, ra đúng như cũ), rồi chép từ Drive về `work/`.

---

## 3. Dữ liệu (GUIDE bước 1) — ≈ 10 phút

```bash
!python scripts/prepare_data.py --kaggle
```

Kiểm tra trong notebook (mục `## 1.`):

- [ ] Số ảnh từng split khớp README: **train 1.422 · val 276 · test 311**.
- [ ] Không `session` nào nằm ở hai split (giao các tập `session` phải rỗng).
- [ ] Vẽ 6 ảnh train có box → hiểu định dạng `yolo = [xc, yc, w, h]` chuẩn hoá [0,1].
- [ ] Ghi `dataset_version` trong `data/splits/meta.json` vào đầu `REPORT.md`.
- [ ] Ghi lại số box từng lớp mỗi split (cho **Câu 6**).

---

## 4. Nhãn và prompt (GUIDE bước 2) — ≈ 15 phút

- Viết `yolo_to_box()` và `objects_to_answer()` đúng pseudo code: số **nguyên**, kẹp [0, 1000], sắp theo `(y1, x1)`, `json.dumps(..., separators=(",", ":"))`, ảnh không có object → `"[]"`.
- Viết **một** hằng `PROMPT` dùng chung cho E0–E4: nêu 3 lớp `pothole`, `crack`, `manhole`; yêu cầu **chỉ** trả JSON list dạng `[{"label": ..., "bbox_2d": [x1,y1,x2,y2]}]`; nói rõ toạ độ **thang 0–1000**; nếu không có gì trả `[]`.
- [ ] Kiểm tra round-trip: JSON → pixel (`x * 640 / 1000`, `y * 360 / 1000`) → vẽ, box trùng box gốc.

Commit: `git add lab_day10.ipynb && git commit -m "step 1-2: data split check + YOLO→JSON 0-1000"`.

---

## 5. E0 zero-shot (GUIDE bước 3) — ≈ 30 phút

1. Nạp model đúng cấu hình 4-bit trong GUIDE (`llm_int8_skip_modules=["visual", "lm_head"]`, `dtype=float16`, `padding_side="left"`).
2. Đặt các hàm `load_model()`, `predict(model, processor, split, out_path)` thành hàm để **tái sử dụng** cho E1–E4.
3. Suy luận **toàn bộ 311 ảnh test**, batch 4, `do_sample=False`, `max_new_tokens=256`. Cắt bỏ phần prompt: `out[:, inputs["input_ids"].shape[1]:]`.
4. Ghi `work/e0.jsonl`, mỗi dòng `{"image_id": ..., "raw": ...}`.
   - **Mọi** `image_id` của test phải có dòng, kể cả khi lỗi (ghi `"raw": ""`) — thiếu dòng làm `missing_predictions > 0` → FAIL.
   - Không trùng `image_id` (score.py sẽ thoát với lỗi).
5. Chấm:

```bash
!python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
```

- [ ] In 5 output thô (cho **Câu 2**): có bọc ```` ```json ```` không, có chữ thừa không, toạ độ thang gì.
- [ ] Ghi `parse_valid_rate`, F1, CI95.

Commit: `step 3: E0 zero-shot evidence`.

---

## 6. Dữ liệu huấn luyện và setup LoRA (GUIDE bước 4–5) — ≈ 30 phút

### 6.1 `encode_example` + `collate`

- [ ] Assert prompt là tiền tố của chuỗi đầy đủ; `labels[:, :n] = -100`.
- [ ] Số token ảnh của ảnh 640×360 = **220**.
- [ ] Decode các token có `labels != -100` → phải đúng JSON đáp án + `<|im_end|>`. **Thiếu `<|im_end|>` = model không học cách dừng** → `parse_valid_rate` thấp.
- [ ] Collate: pad `input_ids`/`attention_mask`/`mm_token_type_ids`/`labels` (labels pad = −100); `pixel_values` và `image_grid_thw` **nối** theo chiều 0, không pad.

### 6.2 `apply_setup(model, setup)`

- [ ] In `[n for n, _ in model.named_modules()]` để xác nhận tên module trên model thật.
- [ ] `len(matched_modules)` = **252** (E1, E2) và **348** (E3).
- [ ] E2/E3: tham số `visual.merger` + `visual.deepstack_merger_list` → `float32`, `requires_grad=True`; số tham số huấn luyện E2 ≈ E1 + **109M**.
- [ ] Ghi số tham số từng khối (LLM ≈ 4,0B, ViT ≈ 415M, merger ≈ 27M, deepstack ≈ 82M) cho **Câu 1**.

Gợi ý regex `target_modules` (peft dùng `re.fullmatch` khi truyền chuỗi):

```python
LLM_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
target = LLM_RE if setup != "E3" else f"({LLM_RE})|({VIT_RE})"
```

Commit: `step 4-5: training data encoding, loss mask, LoRA setups`.

---

## 7. Train + chấm E1, E2, E3 (GUIDE bước 6–7) — ≈ 2 giờ

Viết **một** hàm `train(setup, out_dir, **overrides)` và gọi 3 lần. Siêu tham số lấy từ `configs/e1–e3.yaml` (lr 2e-4, batch 2, grad_accum 4, 300 bước, seed 42) — **không đổi giữa E1–E3**.

### 7.1 Trong vòng train

- [ ] Gradient checkpointing (`use_reentrant=False`), `model.config.use_cache = False`, `enable_input_require_grads()`.
- [ ] `GradScaler` + `autocast(fp16)`; dừng ngay nếu loss không hữu hạn.
- [ ] Đặt `torch.cuda.reset_peak_memory_stats()` **trước** khi train; cuối cùng đo `torch.cuda.max_memory_allocated() / 1e9`.
- [ ] Mỗi 100 bước: `model.save_pretrained(adapter_dir)` **và** lưu projector riêng (E2/E3):

```python
extra = {n: p.detach().cpu() for n, p in model.named_parameters()
         if p.requires_grad and ("visual.merger" in n or "visual.deepstack_merger_list" in n)}
torch.save(extra, f"{out_dir}_extra_trainable.pt")       # .pt bị .gitignore chặn — chép lên Drive
```

- [ ] Ghi `work/eN_run.json`: `{"trainable_params": ..., "peak_vram_gb": ..., "train_minutes": ...}`.
- [ ] Lưu log loss (list) để vẽ đường loss cho báo cáo.

### 7.2 Suy luận

1. **Restart runtime** (hoặc `del model; gc.collect(); torch.cuda.empty_cache()`) để giải phóng VRAM.
2. Nạp model gốc 4-bit → `PeftModel.from_pretrained(model, adapter_dir)` → (E2/E3) `model.load_state_dict(extra, strict=False)`.
   Kiểm tra: `missing/unexpected` của `load_state_dict` — **`unexpected_keys` phải rỗng** (tên key phải khớp tiền tố `base_model.model.…`). Nếu không khớp, F1 E2/E3 sẽ giống hệt E1.
3. Gỡ lỗi trước trên ~40 ảnh `val`:

```bash
!python scripts/score.py --pred work/e1_val.jsonl --split val --name E1-val --out work/e1_val.json
```

   (Thiếu ảnh val là bình thường ở đây — đây chỉ là gỡ lỗi.)
4. Khi ổn, suy luận **toàn bộ test** → `work/e1.jsonl` rồi chấm kèm meta:

```bash
!python scripts/score.py --pred work/e1.jsonl --name E1 --out report/evidence/e1.json \
    --meta trainable_params=<int> --meta peak_vram_gb=<float> --meta train_minutes=<float>
```

   Mẹo: lấy số từ `work/e1_run.json` bằng Python rồi ghép lệnh, tránh gõ tay sai. `peak_vram_gb`/`train_minutes` nên có dấu chấm (vd `9.84`) để được đọc là số thực; `trainable_params` là số nguyên.

5. Lặp lại cho E2, E3. Commit sau **mỗi** setup (`step 6-7: E1 trained + evidence`, …).

### 7.3 Kiểm tra sau E1

- [ ] `parse_valid_rate(E1) ≥ 0,95` và `F1(E1) ≥ F1(E0) + 0,10`. Nếu không: xem lại mask loss (`<|im_end|>` trong nhãn) và thang toạ độ (bảng "Khi gặp sự cố" trong GUIDE).
- [ ] E2/E3 F1 **không** trùng khít E1 (nếu trùng → quên nạp projector).

---

## 8. E4 ablation (GUIDE bước 8) — ≈ 40 phút

Chọn **một** biến, giữ mọi thứ khác như setup tốt nhất E1–E3:

| Biến | Chi phí | Gợi ý |
|---|---|---|
| `rank` (4 vs 64, alpha = 2r) | 2 lần train (setup gốc là r=16, không trùng giá trị nào) | dễ nhất, không đổi pipeline |
| `data_size` (10% vs 100%) | 1 lần train (100% = setup gốc) | rẻ nhất; dùng seed cố định khi lấy 10% |
| `coord_scale` (norm1000 vs pixel) | 1 lần train (norm1000 = setup gốc) | hữu ích nhất cho **Câu 3**; phải đổi **cả** nhãn, prompt và chấm `--coord pixel` |

Đặt tên file: `report/evidence/e4_<biến>_<giá trị>.json`, ví dụ `e4_data_size_0.1.json`, `e4_data_size_1.0.json`. Khi một giá trị trùng setup gốc, chấm lại cùng predictions đó với `--name` mới — **không cần train lại**.

> Mọi file `e4*.json` trong `report/evidence/` đều bị `check_submission.py` đọc: đừng để file val/nháp ở đó.

Mẹo cho **Câu 3** (không tốn GPU): chấm lại `work/e1.jsonl` với `--coord pixel --out work/e1_as_pixel.json` để thấy F1 sụp khi hiểu sai thang. Ghi con số này vào báo cáo.

Commit: `step 8: E4 ablation <biến>`.

---

## 9. E5 GRPO (tuỳ chọn, +10) — chỉ làm khi E0–E4 đã xong

1. Viết `reward()` trong `src/reward.py` và `tests_student/test_reward.py` (thư mục test **riêng**, không đặt vào `tests/` của đề). Đủ 5 ca: hoàn hảo, JSON hỏng, sai lớp, thừa box, rỗng/rỗng. Chạy `python -m pytest tests_student -q`.
2. GRPO với `trl.GRPOTrainer` (`trl==1.9.2`, đọc docs đúng phiên bản), khởi tạo từ adapter tốt nhất, tham số trong `configs/e5.yaml`. Lưu log reward theo bước.
3. Chấm → `report/evidence/e5.json`. Phân tích reward hacking (tỉ lệ output `[]`, box khổng lồ) và so F1 với SFT **kèm CI95**.

---

## 10. Ảnh minh hoạ và báo cáo — ≈ 40 phút

### 10.1 Ảnh (≥ 6, rubric B: 4 điểm)

Từ predictions của setup tốt nhất, chọn ảnh test tiêu biểu: ≥ 2 đúng, ≥ 2 sai (bỏ sót / bịa box / sai lớp), ít nhất một ví dụ cho lớp kém nhất. Vẽ **box đúng (xanh lá)** và **box dự đoán (đỏ, kèm nhãn)** quy về pixel; lưu `report/figures/<tên>.png` và chèn vào `REPORT.md` với chú thích một dòng.

### 10.2 `REPORT.md`

- [ ] Điền họ tên, `dataset_version`.
- [ ] Bảng tổng hợp lấy **trực tiếp** từ `report/evidence/*.json` (sinh bằng script nhỏ để tránh chép sai). Cột: tham số huấn luyện, VRAM đỉnh, phút, F1 (`micro.f1`), CI95 (`f1_ci95`), FP/ảnh (`fp_per_image`), F1 từng lớp (`per_class.*.f1`). VRAM/thời gian phải **khớp `meta`** (rubric B: 4 điểm).
- [ ] 10 câu: trả lời bằng **số của chính bạn**. Câu 3, 4, 6, 8 chấm kỹ nhất:
  - Câu 3: dẫn con số `--coord pixel` vs `norm1000` ở mục 8.
  - Câu 4: nêu tỉ lệ token trả lời / tổng token (220 token ảnh + prompt vs vài chục token JSON).
  - Câu 6: bảng số ảnh/box từng lớp từng split (mục 3).
  - Câu 8: độ rộng CI95 của E1–E3; hai setup có CI chồng nhau thì **không** kết luận được cái nào hơn.
- [ ] Phần "Nhìn lại": lỗi bạn gặp thật sự (OOM, nan, projector không nạp…).
- [ ] `grep -n "<điền" REPORT.md` không còn kết quả.

Commit: `report: results table, figures, answers`.

---

## 11. Trước khi nộp

```bash
python scripts/check_submission.py      # phải: 0 FAIL
git status                              # sạch
git ls-files | grep -E '^(data|work)/|\.env$|\.(safetensors|bin|pt|ckpt)$'   # phải rỗng
git log --oneline | wc -l               # ≥ 5
git push
```

Checklist cuối:

- [ ] `e0`–`e3.json` đủ, `split=test`, `missing_predictions=0`, cùng `dataset_version`.
- [ ] `meta` của E1–E3 đủ 3 khoá, số dương.
- [ ] Ít nhất một `e4*.json` (nên có cả hai giá trị).
- [ ] ≥ 6 ảnh minh hoạ có chú thích.
- [ ] `REPORT.md` đã điền hết, số khớp evidence.
- [ ] Không sửa `scripts/`, `tests/` (`git diff <commit đầu> -- scripts tests` rỗng).
- [ ] Repo **public**, đã push commit cuối **trước hạn**.
- [ ] Sẵn sàng vấn đáp 5 phút: giải thích được từng con số trong báo cáo.

---

## Phụ lục: ngân sách thời gian GPU

| Việc | Ước tính trên T4 |
|---|---|
| Suy luận 311 ảnh (batch 4) | 10–20 phút / setup |
| Train 300 bước | 20–30 phút / setup (E3 chậm và tốn VRAM nhất) |
| Tổng tối thiểu (E0 + 3 train + 4 suy luận + E4) | ≈ 3–4 giờ GPU |

Colab free giới hạn thời lượng GPU mỗi ngày: nên chia làm 2 buổi (buổi 1: E0, E1, E2; buổi 2: E3, E4, báo cáo) và luôn lưu mọi thứ lên Drive.
