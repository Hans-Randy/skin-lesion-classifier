# Split Strategy

## Problem

HAM10000 contains 10,015 images of 7,470 unique lesions (`lesion_id`). Some lesions
appear in multiple images (different magnifications or angles). Splitting randomly
by image would allow different views of the same lesion to appear in both train and
test — inflating reported metrics by leaking lesion identity across splits.

## Decision

**`StratifiedGroupKFold(n_splits=7)` with `groups=lesion_id`, `y=dx`, `seed=42`.**

- Fold 0 → **test**  (~14.3% of images)
- Fold 1 → **val**   (~14.3% of images)
- Folds 2–6 → **train** (~71.4% of images)

Actual row counts after splitting (approximate, exact values in `splits/_meta.json`):

| Split | Images | % |
|-------|-------:|--:|
| train | ~7,160 | ~71.5% |
| val   | ~1,428 | ~14.3% |
| test  | ~1,428 | ~14.3% |

> Note: the plan and README say "70/15/15" — these are the actual fractions the
> splitter produces. No one changed the approach; K=7 is the closest integer that
> gives ~15% test without fractional splits.

## Why this splitter

- `StratifiedGroupKFold` guarantees **no `lesion_id` appears in more than one split**
  (the group constraint) while maintaining **class proportion** across splits (the
  stratify constraint).
- Tested viable for all 7 classes: the smallest class (`df`) has 73 unique lesion
  groups, giving ~10 groups per fold — sufficient for meaningful val/test F1.

## Class distribution per class (groups)

| Class | Total groups | ~val groups | ~test groups |
|-------|------------:|------------:|-------------:|
| nv    | 5,403       | ~772        | ~772         |
| mel   | 614         | ~88         | ~88          |
| bkl   | 727         | ~104        | ~104         |
| bcc   | 327         | ~47         | ~47          |
| akiec | 228         | ~33         | ~33          |
| vasc  | 98          | ~14         | ~14          |
| df    | 73          | ~10         | ~10          |

## Cache contract

Splits are written once to `splits/{train,val,test}.csv` and `splits/_meta.json`.

`_meta.json` records:
```json
{
  "seed": 42,
  "n_splits": 7,
  "test_fold": 0,
  "val_fold": 1,
  "ham_metadata_md5": "<md5 of data/HAM10000_metadata.csv>",
  "train_rows": <int>,
  "val_rows": <int>,
  "test_rows": <int>
}
```

At training start, `train.py` reads `_meta.json` and asserts the stored
`ham_metadata_md5` matches the current `data/HAM10000_metadata.csv`. If it
doesn't match (source file changed), the script refuses to run and asks the user
to regenerate splits with `python scripts/make_splits.py`.

To force-regenerate: `python scripts/make_splits.py --overwrite`.

## Zero-overlap assertion

After generating, the script asserts:

```
len(set(train_ids) & set(val_ids)) == 0
len(set(train_ids) & set(test_ids)) == 0
len(set(val_ids)   & set(test_ids)) == 0
```

where `*_ids` is the set of `lesion_id` values in each split. This runs every
time splits are generated, not just once.

## Files written

| File | Columns |
|------|---------|
| `splits/train.csv` | `image_id`, `dx`, `lesion_id` |
| `splits/val.csv`   | `image_id`, `dx`, `lesion_id` |
| `splits/test.csv`  | `image_id`, `dx`, `lesion_id` |
| `splits/_meta.json`| seed, n_splits, md5, row counts |
