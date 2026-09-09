# Getting the dataset

Source used by the base paper: Kaggle, `ethon0426/lending-club-20072020q1`
https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1

## Steps

1. Create a free Kaggle account if you don't have one.
2. Get your API token: Kaggle profile → Account → "Create New API Token".
   This downloads `kaggle.json` — keep it private, never commit it.
3. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Download into this folder:
   ```bash
   kaggle datasets download -d ethon0426/lending-club-20072020q1 -p data/raw --unzip
   ```

## Notes

- The full file is a few GB — make sure whoever's laptop/Colab is doing the
  initial cleaning has the space and RAM (a Colab Pro instance or a shared
  server is worth setting up now rather than after someone's laptop chokes).
- Do NOT commit the raw CSV to git — it's already covered by .gitignore.
  Once cleaned, commit the trimmed `data/processed/` version if it's small
  enough (or store it via Git LFS / a shared drive if not).
- The base paper reduces 141 raw columns to 57 after removing leaky
  post-approval fields and columns with >40% missing values — replicate this
  exact filtering first (see Phase 1), don't invent a new cleaning process.
