# Night 5 (2026-10-20/21): the "maximum" build, C1T in production, and the rollback

Run: 20:00 → 00:36 UTC on Asgard (2× CMP 170HX, CT 232), test only. The next evening C1T went into production and was rolled back to the night-4 standard after about a quarter of an hour.
- Log: [`../results/logs/night5-NIGHT-LOG.md`](../results/logs/night5-NIGHT-LOG.md).
- Data: speed in [`../results/speed/`](../results/speed/) (`*-n5-*`, `*-N5-prod`) and [`../results/speed/night5/`](../results/speed/night5/), concurrency in [`../results/concurrency/night5-conc.jsonl`](../results/concurrency/night5-conc.jsonl), quality in [`../results/quality/`](../results/quality/) (`n5-*`) with the paired comparison [`compare-night5.md`](../results/quality/compare-night5.md), checks in [`../results/checks/night5/`](../results/checks/night5/), profiles in [`../results/profiles/night5/`](../results/profiles/night5/), scripts in [`../bench/night5/`](../bench/night5/).

## Bottom line

1. **Production stays on the night-4 standard.** On 2026-10-21, C1T served production for about a quarter of an hour: 19:00 to 19:16 UTC, then the rollback.
   - **What C1T is:** the night's pick C3 without probabilistic drafting. That means Swift-A16 (BF16 attention), INT8 per-token-head KV with patch 10 (K1), terse answers as the template default, and the night-4 drafting.
   - **Why it was rolled back:** multi-turn tool use is lower (point 5), and raw decode is 13% slower.
2. **The trade C1T offered,** against the night-4 standard:
   - raw decode 13% slower: 174.6 against 201.6 tok/s in production, default clocks;
   - shorter answers, so tasks finish 28% sooner: 5.46 against 7.56 s mean task latency;
   - a 48% larger KV pool: 1.24M against 838K tokens;
   - better long context: 100% against 95.0%;
   - total tool calls about equal (BFCL 80.8% against 80.5%), but multi-turn tool use lower.
3. **INT8 KV with K1 passed the full-pool test.** 25 unique 72K-token prompts, 1.45× the pool, evicted cached prefixes, and every answer stayed correct.
4. **Terse probably costs about a point of BFCL against A16 without it.**
   - C3 against C1 is −1.4 points (95% CI −2.5 … −0.3, p = 0.018, paired).
   - The terse step alone (C3 against C2) is −0.8 (CI −1.9 … +0.2, p = 0.14), within the noise.
   - Against the night-4 standard, C3 is level.
5. **Multi-turn tool use is lower** (a post-hoc finding).
   - On BFCL's multi-turn subset (100 tasks, 3 runs each), C3 scores 51.7% against 61.3% for R: −9.7 points, paired 95% CI −15.0 … −4.3.
   - 25 tasks got worse and 6 better (sign test p = 0.0009, still under 0.05 after correcting for the 44 subset comparisons).
   - Each step on its own is within the noise: A16 + INT8 KV −4.3, probabilistic drafting −2.3, terse −3.0. Together they are not.

## Setup

Four builds on the same night, clocks locked at 1440 MHz, the same harness:

| Build | What it is | Profile |
|---|---|---|
| **R** | the night-4 standard: Swift 1.5 AWQ, BF16 KV, MTP 4 + n-gram lookup + int4 draft head | [`R.env`](../results/profiles/night5/R.env) |
| **C1** | R + Swift-A16 (BF16 attention) + INT8 per-token-head KV (K1) | [`C1.env`](../results/profiles/night5/C1.env) |
| **C2** | C1 + `draft_sample_method: probabilistic` | [`C2.env`](../results/profiles/night5/C2.env) |
| **C3** | C2 + `terse: true` as the template default | [`C3.env`](../results/profiles/night5/C3.env) |

**Quality.** Every build got the full suite at temperature 1.0 (top-p 0.95, top-k 20, effort medium):
- BFCL v4 three times (3,120 items);
- EvalPlus (542);
- long context (180).

All four runs used the same items and seeds, so the comparison is paired per item.

**Speed.** The night-4 speed set (exact-length prefill, code and code-edit decode at temperature 0.6, func and multi-turn checks). The reference R also got the full prefill matrix from 1K to 128K and the concurrency runs, and so did C3.

**The decision.** [`decide.py`](../bench/night5/decide.py) picks the most complete candidate that passes every gate against R:
- BFCL no more than 1.5 points below R;
- EvalPlus and long context no more than 2 points below R;
- mean task latency at most 1.1× R;
- the func and multi-turn checks, and for INT8 KV the K1 checks.

The night ran with `DEPLOY=0`, so it only reported.

## Results

| Build | BFCL | EvalPlus | Long context | Mean task latency | Code decode | Code edit | KV pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 80.5% | 85.8% | 95.0% | 7.56 s | 191.1 | 304.2 | 838,024 |
| C1 | **82.2%** | 86.0% | **100%** | 7.77 s | 172.3 | 264.3 | 1,239,671 |
| C2 | 81.6% | 85.8% | 100% | 7.83 s | 168.4 | 264.0 | 1,234,771 |
| C3 | 80.8% | 85.6% | 100% | **5.46 s** | 178.6 | 262.8 | 1,234,771 |

- **Sources:**
  - quality: [`compare-night5.md`](../results/quality/compare-night5.md) and `n5-*/medium/*.summary.json`;
  - latency and the decision: [`decide.txt`](../results/speed/night5/decide.txt);
  - decode: `decode-n5-*.json` and `speed/night5/edit-*.txt`;
  - KV pools: the night log.
- **Units and settings:** tok/s at 1440 MHz, temperature 0.6. Code edit is with thinking off.
- **Mean task latency** is the mean of the median per-task latencies of BFCL and EvalPlus at T=1. It is what a user waits for an answer.

**Paired comparisons** ([`compare-night5.md`](../results/quality/compare-night5.md); item-clustered bootstrap and sign test):

| Pair | BFCL | EvalPlus | Long context |
|---|---|---|---|
| C1 − R (A16 + INT8 KV) | **+1.6** (CI +0.3 … +2.9, p = 0.023) | +0.2 (n.s.) | **+5.0** (CI +2.2 … +7.8, p = 0.004) |
| C2 − C1 (probabilistic drafting) | −0.5 (CI −1.5 … +0.4, n.s.) | −0.2 (n.s.) | 0 |
| C3 − C2 (terse) | −0.8 (CI −1.9 … +0.2, p = 0.14) | −0.2 (n.s.) | 0 |
| C3 − C1 (both) | **−1.4** (CI −2.5 … −0.3, p = 0.018) | −0.4 (n.s.) | 0 |
| C3 − R | +0.3 (CI −1.1 … +1.6, n.s.) | −0.2 (n.s.) | **+5.0** |

**What each step did:**
- **A16 + INT8 KV (C1).** It repeats night 4's A16 gain on tool calls (+1.6) and long context (+5.0; R missed 9 of 180 needles, C1 none). Code is unchanged. Raw decode costs 10% at 1440 MHz, and code edits 13%. The KV pool grows from 838K to 1.24M tokens: INT8 KV more than wins back what the BF16 attention weights take.
- **Probabilistic drafting (C2): no gain**, on quality or on task latency (7.83 against 7.77 s). It is left out.
  - Night 4's decode-at-size gain does not show up in what users wait for.
  - The night's speed tools sample at temperature 0.6, so a decode-speed comparison at T=1 still does not exist.
- **Terse (C3): tasks finish 28% sooner than with R** (5.46 against 7.56 s) and 30% sooner than with C2.
  - Answers use 17% fewer tokens on BFCL (346 against 418 for R) and 48% fewer on EvalPlus (491 against 936).
  - EvalPlus is unchanged. BFCL is −0.8 against C2, within the noise, but −1.4 against C1, outside it.
  - On the BFCL subsets, terse helps the irrelevance categories (89.3% against 84.3% for C2) and costs on simple, multiple and multi-turn calls; see the subset table in the comparison.

**Multi-turn tool use.** BFCL's `multi_turn_base` subset falls step by step: R 61.3%, C1 57.0%, C2 54.7%, C3 51.7%.
- **How it was computed:** per task, the mean over the three runs, compared between builds on the same 100 tasks (paired bootstrap, 10,000 resamples), from the published per-item files `n5-*/medium/bfcl.jsonl.gz`.
- **Result:**
  - C3 − R = −9.7 points (95% CI −15.0 … −4.3); 25 tasks got worse and 6 better (sign test p = 0.0009).
  - The single steps are within the noise: C1 − R −4.3 (CI −10.0 … +1.0), C2 − C1 −2.3, C3 − C2 −3.0.
- **Caveat:** this is one subset out of 11, looked at after the fact. It survives a Bonferroni correction over 11 subsets × 4 pairs (p = 0.04), but night 4 did not show it: A16 56.5% against 58.0% for the reference, and terse 58.0% on one run.
- **Why it matters:** multi-turn tool use is the closest BFCL gets to agentic coding. It is why C1T was rolled back, and it is the first thing to measure on any build before it goes into production.

**decide.py picked C3:** all three candidates passed every gate, and C3 is the most complete.

## INT8 KV with K1: the full-pool test

C1 ran every K1 check before its speed and quality runs ([`../results/checks/night5/`](../results/checks/night5/), `C1-*`). All passed, and the server log shows 8 copy-on-write steps, 0 skipped scale copies and 0 tracebacks:

| Check | Result |
|---|---|
| first batch after start, 12 users (`C1-first`) | all pass |
| multi-turn 6 × 5 with cold controls (`C1-mt`) | all pass; 22 of 24 follow-up hits ended inside a block; median follow-up TTFT 0.17 s |
| fanout, 16 users on one cached document (`C1-fan`) | all pass |
| **full pool** (`C1-pinned`) | all 12 answers correct, see below |
| multi-turn again after the flood (`C1-mt2`) | all pass |

**The full-pool test:**
- **Setup.** Four conversations kept asking follow-ups while 25 unique prompts of 72,000 tokens, 20 at a time, flooded a 1,239,671-token pool (1.45×).
- **The pool turned over.** Only 78,624 of the 2,070,006 queried tokens were cache hits, so cached prefixes were evicted and reused.
- **Turns during the flood.** Two of them recomputed their evicted prefix: TTFT 2.5 and 2.7 s, as cold as their first turns. The other two waited behind the flood: TTFT 32 and 175 s.
- **Turns after the flood** hit the cache again (0.16–0.18 s).
- **Every answer was correct.**
- **The "peak KV usage 17%" line** in the output counts only the blocks held by running requests. Cached prefixes in free blocks are not counted, which is why it stayed low while the pool turned over.

## C1T in production, and the rollback

On 2026-10-21 the night-4 standard was replaced by **C1T** ([`C1T.env`](../results/profiles/night5/C1T.env)): C3 without probabilistic drafting. C1T itself did not run the quality suite. C3's numbers stand for it, because probabilistic drafting changed nothing measurable (C2 against C1).

Deploy at 18:56–19:04 UTC, with production clocks at default (log lines at the end of the night log):
- **Gate.** `prod_check`, `func_check` and multi-turn passed (80% of the queried tokens were prefix hits), and so did the live feature check (29/29, [`feature-prod.txt`](../results/checks/night5/feature-prod.txt)).
- **KV pool: 1,239,671 tokens.**
  - Source: the service log at start-up, [`prod-start-2026-10-21.txt`](../results/checks/night5/prod-start-2026-10-21.txt).
  - The deploy line in the night log says 838,024. That is wrong: `deploy5.sh` took the first KV-pool line of the service log, from the 00:34 start of the old profile.
- **Code decode 174.6 tok/s** median, acceptance length 3.61 ([`decode-N5-prod.json`](../results/speed/decode-N5-prod.json)). The night-4 standard measured 201.6 the same way ([`decode-N4-prod.json`](../results/speed/decode-N4-prod.json)): −13%.
- **Prefill** ([`exact-N5-prod.json`](../results/speed/exact-N5-prod.json)):
  - 32K: 7,893 tok/s, 101% of night 4;
  - 4K: 4,709 tok/s, 95%. The terse instruction makes the 4K test prompt about 155 tokens longer (4,265 against 4,110).

**Concurrency** was measured on C3, which drafts probabilistically, at 1440 MHz:

| | C3 | R |
|---|---:|---:|
| 8 users | 359 tok/s | 404 tok/s |
| 16 users | 440 tok/s | 511 tok/s |
| 4 chat + 1 agent at 33K | 162 tok/s | 168 tok/s |

- The chat p95 TTFT in the mixed run was 1.83 s for C3 against 2.25 s for R.
- Total tok/s falls by about as much as single-stream decode does.
- C1T itself has not been measured under load.

**The rollback.** At 19:16 UTC production went back to the night-4 standard ([`../config/qfn-prod.env`](../config/qfn-prod.env)), because of two costs:
- the multi-turn drop (C3 −9.7 points against R), which matters most for agentic coding;
- 13% less raw decode.

C1T had been live since 19:00 UTC. The rollback ([`rollback-2026-10-21.txt`](../results/checks/night5/rollback-2026-10-21.txt), script [`rollback-c1t.sh`](../bench/night5/rollback-c1t.sh)):
- **19:15:59:** started.
- **19:19:55:** the standard served again, with BF16 KV and an 838,024-token pool.
- **Then:** `prod_check`, `func_check` and multi-turn passed.
- **19:22:10:** done. Code decode was 194.7 tok/s median, acceptance length 3.59, at default clocks ([`decode-N5-rollback.json`](../results/speed/decode-N5-rollback.json)). The night-4 deploy had measured 201.6 the same way. That difference is within the run-to-run spread: the same profile gave 191.1 and 200.8 in two runs during night 5.

INT8 KV is off again: patch 10 (K1) stays in the tree, tested correct, and does nothing with BF16 KV.

## Next

Test INT8 KV alone on the standard (Swift 1.5 AWQ, the same drafting, no terse) with BFCL three times, the multi-turn subset included, before using it.
- The +71% KV pool (night 4) is the reason to want it.
- C1's multi-turn result (57.0% against 61.3%) is the reason to check first. C1 also had BF16 attention, so night 5 cannot say whether INT8 KV or A16 caused it.

## Open

- A decode-speed comparison at T=1, the setting the quality suites and Qwen's model card use.
- Whether the multi-turn drop repeats, and whether it comes from A16, INT8 KV or terse.
- Whether terse's cost on BFCL against C1 (−1.4) repeats.
- `deploy5.sh`: take the last KV-pool line of the service log instead of the first.
