# Night 4 (2026-10-19/20): drafters, Q6 quality builds, DFlash2, INT8 KV — results

Run: 20:00 → 03:22 UTC on Asgard (2× CMP 170HX, CT 232).
- A/B phases ran with the clocks locked at 1440 MHz.
- Log: `/root/tune/night4/NIGHT-LOG.md` (in this repository: [`../results/logs/night4-NIGHT-LOG.md`](../results/logs/night4-NIGHT-LOG.md)).
- Data in this repository: speed in [`../results/speed/`](../results/speed/) (`*-n4-*`, `*-N4-prod`), checks in [`../results/checks/night4/`](../results/checks/night4/), quality in [`../results/quality/`](../results/quality/) (`n4-*`), profiles in [`../results/profiles/night4/`](../results/profiles/night4/), patches 08–18 in [`../patches/`](../patches/), tools in [`../bench/night4/`](../bench/night4/) and [`../tools/models/`](../tools/models/).

## Bottom line

1. **Deployed: int4 draft head + MTP depth 4 + prefix-match unit 16.** Production code decode is **201.6 tok/s, up from 169 (+19%)**, and code-edit decode is about +22%. Prefill is unchanged.
   - Quality is unchanged on the same night: BFCL 79.6 / EvalPlus 86.0 / long context 98.3.
   - The gate passed (prod, func, multiturn), and features are 29/29.
2. **Swift-A16 (BF16 attention) is the quality option.**
   - Quality: BFCL **+2.6 pt** (82.1) and long context **+1.1 pt** (99.4), with EvalPlus the same.
   - Cost: −10% decode. It is still faster than Intel.
3. **Q6 answer: stop at A16.** INT8 experts in 16 layers (Q6x) measured the same as A16 and cost a further 4% decode and 40% of the KV pool. The quality gain comes from attention precision, not expert precision.
4. **INT8 KV is fixed (patch K1).**
   - The control reproduced the old bug: 10/28 cache-hit turns failed.
   - With K1, every test passed.
   - Gain: **+71% KV pool** (838K → 1,433K tokens) for −1–2% decode, with long context unchanged (98.3).
5. **DFlash2 works on our stack:**
   - lossless;
   - multi-user safe;
   - acceptance 3.0 at 3 drafts and 3.3 at 4 drafts.

   It does not beat our MTP4 setup, though: code is −2% at 4 drafts and code-edit −24%, because it cannot use n-gram lookup.
6. **Probabilistic drafting:**
   - +13–18% decode at T=1 (the model's recommended sampling), and no gain at T=0.6;
   - *Note added on publication:* the speed tools (`perf_exact.py`, `decode_code.py`, `edit_decode.py`) all sample at temperature 0.6. The +13–18% is the 8K/32K decode-at-size of `perf_exact.py` (3 runs of 256 tokens), so it was measured at T=0.6, not T=1; code decode (−3%) and code edit (±0) at the same temperature show no gain. A real T=1 speed comparison is still owed;
   - no measurable bias;
   - BFCL 81.4.
7. **Terse:** same quality (BFCL 81.5, EvalPlus 85.8), with **−16% tokens on BFCL and −43% on EvalPlus**. EvalPlus median latency halves (9.9 → 5.0 s).

## Deployed now (production 8080)

**Profile:** `/opt/vllm-qfn/qfn-prod.env`.
- Venv `/opt/vllm-qfn-n4` (the patched tree), model `swift15-awq`.
- `VLLM_MTP_DRAFT_HEAD_PATH=…/swift15-awq-lm_head-w4a16.safetensors`.
- MTP 4 + n-gram lookup.
- `--prefix-match-unit 16`. With 4 drafts the attention block becomes 1616 tokens, which is not divisible by 32.

**Rollback:** `/root/tune/night4/qfn-prod.env.before-deploy`, or `qfn-prod.env.baseline`. Copy it to `/opt/vllm-qfn/qfn-prod.env` and run `systemctl restart vllm-qfn`.

**Patches active in production** (the rest are inactive unless their switch is set):
- D1 int4 draft head;
- C1 reject never-proposed drafts;
- C2 no spec padding of fresh prompts.

| On 8080, clocks unlocked | Night 3 | Night 4 |
|---|---:|---:|
| Code decode (median) | 169 | **201.6** (min 176.5, max 215.9) |
| Acceptance length | ~3.1 | **3.62** |
| Prefill 4K / 32K | 4,920 / 7,788 | 4,944 / 7,838 |

## Speed: drafters (clocks 1440 MHz, same harness)

| Config | Code tok/s | Edit tok/s | Accept | KV pool | Notes |
|---|---:|---:|---:|---:|---|
| T0 baseline (MTP3 + n-gram) | 168.5 | 248.0 | 3.12 | 877K | production before tonight |
| No drafting | 83.0 | 83.9 | 1.00 | 1,497K | MTP+n-gram roughly doubles decode |
| D3 int4 head | 179.2 | 256.8 | 3.05 | 857K | +4.9% |
| **D4p16 int4 head + MTP4** | **193.6** | **303.7** | **3.51** | 838K | **+18.6% → deployed** |
| MTP5 | – | – | – | – | cannot start: the QSA ring (12) must divide block 1616 |
| D4p16 + INT4 MTP experts | 186.8 | 307.9 | 3.42 | **1,054K** | −1.1%; rejected by the 1% rule, +26% KV |
| D4p16 + probabilistic drafting | 187.9 | 303.6 | 3.51 | 835K | at T=1: 8K decode 181 vs 153, 32K 171 vs 151 (measured at T=0.6, see the note under Bottom line) |
| DFlash2, 3 drafts | 178.2 | 214.7 | 3.02 | 824K | report only |
| DFlash2, 4 drafts | 189.0 | 231.9 | 3.31 | 805K | report only |

**Batch size and stage split** (prefill tok/s at 8K / 32K / 128K):

| Variant | Prefill | Decode | KV pool | Result |
|---|---|---|---:|---|
| Reference | 5,733 / 7,696 / 8,169 | – | – | kept |
| Batch 3296 | ±0 | – | – | no gain |
| Batch 4896 | 128K +5.6%, 8K −11% | – | – | rejected |
| Split 26/22 | −3% | −3% | **1,004K (+20%)** | rejected for speed |

**Bounded prefill under mixed load** (5 users, agent + chat):

| Setting | Chat TTFT p95 | Agent TTFT median | Throughput |
|---|---:|---:|---:|
| Off (current) | 0.45 s | 4.9 s | 171 tok/s |
| Stock threshold 3200 | 1.43 s | 5.8 s | 160 |
| D2 at 1600 | 5.57 s | 7.2 s | 136 |

At 8 users all three are equal at about 394 tok/s. Keep it off.

## Quality (same night, same harness, effort medium)

| Build | BFCL | EvalPlus | Long context | Code tok/s | KV pool |
|---|---:|---:|---:|---:|---:|
| Swift, deploy candidate (INT4 attention) | 79.6 (2,080) | 86.0 | 98.3 | 193.6 | 838K |
| **Swift-A16** (BF16 attention) | **82.1** (2,080) | 85.8 | **99.4** | 173.7 | 724K |
| Swift-Q6x (A16 + INT8 experts in layers 0–10 and 43–47) | 81.5 (1,264, time budget) | 86.3 | 100 | 166.8 | 443K |
| Intel AutoRound + int4 head + INT4 MTP | (+3 pt vs Swift on earlier nights) | | | 160.8 | 936K |
| Probabilistic drafting | 81.4 (1,040) | – | – | | |
| Terse | 81.5 (1,040) | 85.8 | – | | |
| INT8 KV + K1 | – | – | 98.3 | 190.4 | 1,433K |

Night 3 production reference: 80.3 / 86.5 / 97.2. BFCL resolves about ±2 pt.

**Terse tokens per answer:**

| | Normal | Terse |
|---|---:|---:|
| BFCL | 421 | 354 |
| EvalPlus | 933 | 536 |
| EvalPlus latency median | 9.9 s | 5.0 s |

## Correctness checks

**Drafting kernels and sampling:**
- **T3 kernel exactness:** the drafting and rejection kernels are unbiased. The old shared-noise bug is flagged (TV 0.066), so the test can see it. The C1 unit check passes.
- **Bias test** (1,500 samples, letters and digits): probabilistic drafting matches greedy drafting. TV 0.057 / 0.041 vs 0.054 / 0.039. No added bias.

**Canaries:**
- **131K-prompt preemption canary:** 0 preemptions.
- **T1 dead drafter** (1/3/5/6/7 users): all 21 streams kept drafting for T0, probabilistic and DFlash2.
- **T2 fresh-prompt probe:** 24/24 with C2. It also passed 24/24 with stock padding, so the padding bug does not reproduce here and C2 is a precaution.
- **T4 prefix-bleed soak:** 72/72 and 36/36.
- **C4 PP-skew probe:** it fired only on the 24 warm-up rows at startup, and never during traffic, so the vektorprime fix is not needed.

**Greedy identity:**
- Results: 5–10 of 16 answers identical across every drafter change, including no drafting vs MTP3 (10/16).
- **Why:** the divergences are near-tie flips about 200 characters in. The verify step runs 1+k tokens per step, so the W4A16 kernels reduce in a different order.
- **Not a lossless test here:** this check cannot prove losslessness on this setup. The quality suites are the real check, and they showed no loss.

**K1 (INT8 KV):**
- **Control** (`VLLM_K1_DISABLE=1`): 10/28 cache-hit turns failed, 74 copy-on-write steps had their scale copy skipped, and 598 scale rows were stale.
- **With K1:** multiturn 28/28, fanout (12 users) passed, first batch passed, multiturn 6×5 passed.
- **Server log:** "NOT copied" 0, partial copies 0, tracebacks 0. Follow-up TTFT 0.17 s.
- **Not stressed:** eviction with a full pool. The churn test only reached 17% KV usage.

## DFlash2 (incoai/Qwen3.8-Flash-Next-DFlash2)

**What it took to run:** three patches and three attempts.
- **D3:** a Flash-Next feature tap for layers 5/15/25/35/45, HC-mean, with PP0→PP1 relay. The loader's PP filter is turned off for the drafter.
- **D3b:** `dflash` added to the Qwen4Exp spec-method allow-list.
- **D3c:** a guard for the missing MTP buffer in the runner.

**Proof the tap works:** acceptance is 2.97–3.31; feature mistakes would give about 1.

**Results:**
- 3 drafts vs the old MTP3+n-gram: code +6%, edit −13%.
- 4 drafts vs our MTP4: code −2%, edit −24%.
- Short prefill is slower: 1K −22%.

**Verdict:** not for us unless n-gram lookup can be combined with it. That's a possible future port; vLLM currently allows n-gram only with MTP.

## Incidents during the night, and the fixes made

- **Launcher.** The launcher waited the full 20-minute health timeout after an engine died, which cost 16 minutes once. `qfn-window.sh` now gives up as soon as the unit exits or the log shows "Engine core initialization failed". Backup: `night4/qfn-window.sh.pre-n4fix`.
- **DFlash2.** The phase failed twice, on the guard and then on the MTP buffer. It was rerun inside the deploy step after patches D3b and D3c.
- **Quality guard.** Added to `deploy4.sh`. It refuses the deploy if same-night quality falls below night 3 by more than 2.5 / 3 / 3 pt.
- **Log watch.** My first log watch buffered (`cut`) and missed the first 30 minutes of events. No effect on the run.

## Decisions for you

1. **INT8 KV + K1 in production. Recommended.**
   - **Gain:** +71% KV (877K → about 1.5M tokens with the deployed config) for −1–2% decode, with long context unchanged.
   - **Change:** `KV_DTYPE=int8_per_token_head` in the profile. K1 is already in the deployed tree.
   - **Open:** a pool-full eviction test is still owed.
2. **Probabilistic drafting. Recommended if clients use T≈1, as Qwen recommends.** +13–18% decode at T=1, nothing at T=0.6, no measured bias (see the note under Bottom line on the temperature). Change: `"draft_sample_method":"probabilistic"` in the speculative config.
3. **Terse as default.** Same accuracy, −16 to −43% tokens, about half the latency on coding tasks. It is a template default (`terse: true`).
4. **Quality profile.** Swift-A16, optionally with INT8 KV to win back its KV (724K → about 1.2M). The trade is +2.6 BFCL / +1.1 long context against −10% decode. Choose per use: agentic or tool-heavy → A16; raw speed → Swift.
5. **Capacity options, only if KV still limits after item 1:**
   - INT4 MTP drafter: +26% KV, −1%.
   - Split 26/22: +20% KV, −3%.
6. **Cleanup when you're happy.**
   - Temporary: `.q6x-cache` (38 GB) and `.a16-work` (5 GB).
   - Keep or drop: `swift15-q6x` (about 50 GB new) and `dflash2-flashnext` (1.2 GB).

## Suggested night 5

A combined candidate:
- D4p16 + INT8 KV + probabilistic drafting + terse;
- the same build on A16.

Plus the remaining checks:
- the K1 pool-full eviction test;
- greedy determinism (T0 vs T0) to set the identity floor;
- optionally, combining DFlash2 with n-gram lookup.
