import pandas as pd

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ML.config import (  # type: ignore
        PROCESSING_INPUT_PATH,
        PROCESSING_OUTPUT_PATH,
        PROCESSING_TARGET_ROWS,
        SAMPLING_SEED_ADJUSTMENT,
        SAMPLING_SEED_PRIMARY,
    )
else:
    from .config import (
        PROCESSING_INPUT_PATH,
        PROCESSING_OUTPUT_PATH,
        PROCESSING_TARGET_ROWS,
        SAMPLING_SEED_ADJUSTMENT,
        SAMPLING_SEED_PRIMARY,
    )

# ==========================================
# ĐỌC DỮ LIỆU GỐC
# ==========================================
INPUT_PATH = PROCESSING_INPUT_PATH
OUTPUT_PATH = PROCESSING_OUTPUT_PATH
TARGET_ROWS = PROCESSING_TARGET_ROWS
SEED_PRIMARY = SAMPLING_SEED_PRIMARY
SEED_ADJUSTMENT = SAMPLING_SEED_ADJUSTMENT

df = pd.read_csv(INPUT_PATH)
print(f"Số dòng gốc  : {len(df):,}")
print(f"\nPhân phối review_score (gốc):")
orig_dist = df["review_score"].value_counts(normalize=True).sort_index()
print(orig_dist.apply(lambda x: f"{x:.2%}"))

# ==========================================
# STRATIFIED SAMPLING – giữ tỉ lệ review_score
# ==========================================
df_sampled = (
    df.groupby("review_score", group_keys=False)
    .apply(lambda g: g.sample(
        n=max(1, round(len(g) / len(df) * TARGET_ROWS)),
        random_state=SEED_PRIMARY
    ))
)

# Nếu do làm tròn mà lệch so với mục tiêu thì điều chỉnh
# (thêm hoặc bớt ngẫu nhiên không phân biệt class)
diff = TARGET_ROWS - len(df_sampled)
if diff > 0:                              # thiếu → lấy thêm từ phần chưa chọn
    remaining = df.drop(df_sampled.index)
    extra = remaining.sample(n=diff, random_state=SEED_ADJUSTMENT)
    df_sampled = pd.concat([df_sampled, extra])
elif diff < 0:                            # thừa → bỏ ngẫu nhiên
    df_sampled = df_sampled.sample(n=TARGET_ROWS, random_state=SEED_ADJUSTMENT)

df_sampled = df_sampled.sample(frac=1, random_state=SEED_PRIMARY).reset_index(drop=True)

# ==========================================
# KIỂM TRA VÀ LƯU KẾT QUẢ
# ==========================================
print(f"\nSố dòng sau lấy mẫu: {len(df_sampled):,}")
print(f"\nPhân phối review_score (sau lấy mẫu):")
new_dist = df_sampled["review_score"].value_counts(normalize=True).sort_index()
print(new_dist.apply(lambda x: f"{x:.2%}"))

print(f"\nSo sánh tỉ lệ (gốc vs sau lấy mẫu):")
comparison = pd.DataFrame({
    "Gốc (%)":        (orig_dist * 100).round(2),
    "Sau lấy mẫu (%)": (new_dist  * 100).round(2),
    "Chênh lệch (%)": ((new_dist - orig_dist).abs() * 100).round(2),
})
print(comparison.to_string())

df_sampled.to_csv(OUTPUT_PATH, index=False)
print(f"\nĐã lưu file: {OUTPUT_PATH}")
