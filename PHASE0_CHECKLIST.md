# Phase 0 checklist — Week 1

Goal: everyone has a working environment, the dataset, has read the base
paper, and knows their lane. Nothing modelling-related happens this week —
that's the point.

## 1. Kickoff meeting (30-45 min, do this first)

- [ ] Everyone has read the base paper (Monje et al., 2025) in full —
      not skimmed. Each person should be able to state, in one sentence,
      what the base paper's surrogate model actually outputs.
- [ ] Assign rough ownership of the six areas in the README table. Loose is
      fine — these will blur, but someone should be the "default owner" of
      each so nothing falls through the cracks.
- [ ] Agree on communication: one channel (WhatsApp/Slack/Discord) for daily
      chat, one recurring weekly sync slot.
- [ ] Agree on where code lives (GitHub repo — create it now if you haven't)
      and where the compute happens (Colab Pro / college server / a shared
      machine — decide based on who has access to what).

## 2. Repo and environment (each person, same day)

- [ ] Clone the repo.
- [ ] Set up the virtual environment from `requirements.txt`.
- [ ] Confirm `import xgboost, torch, shap` all work without errors.

## 3. Dataset (whoever owns "baseline reproduction" leads this)

- [ ] Download the dataset per `data/raw/README.md`.
- [ ] Load it once, print `.shape` and `.columns`, confirm it matches the
      base paper's numbers before cleaning: 2,925,493 rows, 141 columns.
- [ ] Share the raw file with the team (shared drive link, or everyone
      downloads their own copy — don't commit it to git).

## 4. Reading list (each person picks at least one, report back briefly)

Beyond the base paper, skim one of these so the team isn't relying on a
single person's understanding of the wider literature:

- [ ] A paper on transaction/behavioural-feature credit scoring (open banking
      style) — ground your feature engineering against what's already
      published so you know what's novel vs. what's re-inventing a known
      indicator.
- [ ] A paper on temporal models for credit risk (LSTM/TFT/time-series
      classification on account data) — e.g. work comparing interval-based
      time-series methods against feature-engineered monthly aggregates.
- [ ] A paper on SHAP or surrogate-model explainability in finance.

## 5. End-of-week deliverable

A short team message (not a formal document) answering:
1. Does everyone have the dataset and a working environment?
2. Who owns which of the six areas?
3. One thing each person learned from their reading that's relevant to our
   design choices.

If all three are answered, you're ready to start Phase 1 (baseline
reproduction) next week.
