# siRNA efficacy benchmarks

The four benchmark groups used in the BioPrior / OligoFormer experiments, one CSV
per dataset. These are the **paper-preprocessed** versions (same rows and values
as `data/{Hu,Taka,Mix,Shabalina}.csv`), with two extra provenance columns appended.

| File | Group | Originating study | Assay / protocol | Rows | High-efficacy (y=1) |
|------|-------|-------------------|------------------|-----:|--------------------:|
| `hu.csv`        | Hu        | Huesken (2005), H1299          | mRNA-level          | 2361 | 1174 |
| `taka.csv`      | Taka      | Takayuki (2007), HeLa          | luciferase reporter | 702  | 176  |
| `mix.csv`       | Mix       | 7 studies pooled¹              | mixed (mRNA-level)  | 472  | 241  |
| `shabalina.csv` | Shabalina | Shabalina (2006), multiple     | mRNA-level          | 653  | 289  |

¹ Mix = Reynolds, Vickers, Harborth, Ui-Tei, Khvorova, Hsieh, Amarzguioui.

## Columns

| Column | Meaning |
|--------|---------|
| **`siRNA`** | siRNA **guide sequence**, 19 nt, RNA alphabet (A/C/G/U). **This is the guide column.** |
| `mRNA`   | Local mRNA target context around the site (`X` = padding outside the transcript). |
| **`label`** | **Efficacy value**, min–max normalized to **[0,1]** per dataset. This is the continuous target. |
| **`y`**  | Binary **high-efficacy label** as used in the paper (1 = high efficacy). See caveat below. |
| `source` | Originating study / group (constant within each file). |
| `assay`  | Assay/protocol characterization from the paper (constant within each file). |
| `td`     | 24-dim thermodynamic feature vector (comma-joined, quoted). Not needed for basic use; kept for fidelity. |

**Guide sequence → `siRNA`. Efficacy → `label`.**

## ⚠️ Threshold caveat (read before recomputing labels)

`y` was thresholded on the efficacy **before** the final [0,1] normalization
(OligoFormer's per-study scale, threshold ≈ 0.7). Re-thresholding the `label`
column here at 0.7 does **not** reproduce `y`:

- For Hu the y=0/y=1 boundary sits near `label ≈ 0.52`, not 0.70.
- Mix pools seven studies thresholded individually, so no single cutoff on `label` is exact.

**Use the `y` column directly** for the high-efficacy label. If you want to
re-derive a threshold on a common normalized scale, do it yourself on `label`,
but know it will differ from the paper's `y`.

## Assay / protocol notes (`source`, `assay`)

- The `assay` values are the **paper's group-level characterization** (see the repo
  README's protocol diagnostic): Taka is a single-target **luciferase reporter** assay;
  Hu, Mix, and Shabalina are **mRNA-level** assays. They are **constant per file**, not
  independently re-annotated per row.
- **Mix is heterogeneous** — it pools seven studies with differing cell lines and assays.
  The consolidated file does not retain which of the seven each row came from; `source`/`assay`
  only mark it as the pooled Mix group. If you need per-study Mix provenance, it is not in these files.
- For the **train-on-Hu / test-on-Taka** cross-protocol comparison, both Hu and Taka are
  homogeneous single-study groups, so the constant `source`/`assay` tag is exact for that use.
