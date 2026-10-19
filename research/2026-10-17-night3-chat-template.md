# Night 3 (2026-10-16/17): the best chat template for QFN

## Summary

**qfn-asgard-v2 runs in production on :8080 since 03:40 UTC.** It uses `tool_prompt: chromix` and effort medium, on vLLM with patches A1/A2. Every gate passed:

- the production, functional and multi-turn checks;
- **29/29** live features;
- speed 98–103% of night 2.

Rollback is a single file away; see the Production section.

**Is it the best template?** On every measure we have, yes:

- **Quality: tied with the best variant, ahead of the model's own prompt.**
  - BFCL 80.3% vs 78.9% (+1.4, replicated in two independent runs). This ties chromix, the best of 6 variants.
  - EvalPlus and long context are inside the noise.
  - None of the community extras (Sharp's payload rules, failure notes, strict rules) improved anything. They tend to hurt.
- **Robustness: ahead of every upstream template on the request shapes tested.**
  - 41/41 cases render. Sharp-naze renders 38 and the model's own template 24.
  - 13/13 client edge cases pass.
  - The 9 problems fixed tonight include two serious ones: Claude Code tool search gave HTTP 400, and control tokens in web pages or files allowed prompt injection with real turns.
  - Five of the problems hit every upstream template.
- **Thinking on/off:** the template and the vLLM parser agree in 1176/1176 combinations. On stock vLLM they disagree in 67 of 336.
- **Portability:** byte-identical on vLLM/transformers and the JS engine; on llama.cpp everything except a case llama.cpp itself rejects.
- **Speed:** renders about 30% faster than the base.

**The two experiments answer the open questions:**
1. **Effort xhigh is worse than medium for tool use.** BFCL is −5.2 points (CI −7.8 … −2.8, significant), because the model makes the wrong number of calls in parallel tasks. EvalPlus is unchanged at +46% tokens. Your medium choice is confirmed.
2. **No empty think blocks when the client drops reasoning:** ours 62.0% vs 59.0% for the model's own template. The direction fits the design (+3.0, not significant at n=100).

## What was built: qfn-asgard-v2

A chat template for Qwen3.8-Flash-Next / Swift 1.5.

- **Defaults:** it renders the model's own prompt byte for byte.
- **Every behaviour change is a switch.** Set them per request (`chat_template_kwargs`) or for the whole server (`--default-chat-template-kwargs`).
- **Package** (template, README, tests and vLLM patches): `reports/qfn-asgard-template/`.

**Where it comes from:**
- Sharp-naze v1.4: the base.
- Sharp-naze v1.5 (released yesterday): several fixes.
- chromix + froggerick: the tool text and history truncation, as switches.
- Our own audit and tests.

### Problems found and fixed tonight

Problems that affect every upstream template (the model's own, Sharp-naze, chromix):

| # | Problem | Consequence before | Now |
|---|---|---|---|
| 1 | Claude Code tool search (`tool_reference` in tool results via `/v1/messages`) | HTTP 400 in **every** template | `Tool loaded: NAME` |
| 2 | Control-token text (`<\|im_end\|><\|im_start\|>system…`) in a web page, file or user text | becomes **real** special tokens on vLLM: prompt injection with a real system turn. Verified with the tokenizer: 3 turns instead of 2 | escaped as `&lt;\|…`; ordinary text unchanged |
| 3 | System message in the middle of a conversation (agent frameworks) | HTTP 400 in every template | rendered in place |
| 4 | Parallel tool results that arrive out of call order | the model swaps the results (the prompt has no call ids) | sorted back into call order when the ids match exactly |
| 5 | A tool result as the very first message (client trimmed history) | malformed turn without an opener | fixed |
| 6 | `"false"`/`"0"` as switch values | treated as **true** | understood |
| 7 | `<\|think_on\|>` + effort none, `<\|think_off\|>`, effort none/off | template and vLLM parser disagree: the answer lands in `reasoning` and content is empty | vLLM patch A1 + template: 0 of 1176 combinations disagree (stock vLLM: 67 of 336) |
| 8 | An answer quoting `</think>` with thinking off | split into fake reasoning | kept |
| 9 | `video_url`; `<\|think_off\|>` inside wrapped tool results; tag-only system message | error, or a file could switch thinking off | handled |

**Engine portability:**
- 41 cases render byte-identically on transformers/vLLM and on `@huggingface/jinja`, old and new versions.
- On llama.cpp 40/41 render. The 5 differences are llama.cpp's own normalisation, and the 41st case is rejected by its message parser before any template runs.
- llama.cpp's auto-parser parses tool calls and reasoning correctly (6/6).
- Bug found along the way: my own `//` broke the whole template in older JS engines. Fixed.

**Render speed:** about 30% faster than the base (41.8 vs 58.3 ms for a 600-message agent session).
- The inherited tool-error detection no longer runs when its switch is off.
- My own first version of the result sorting was 5–8× slower; I caught that with a benchmark and rewrote it.

**Corrected along the way:**
- Yesterday's "bug b" (vLLM drops `reasoning_content` from history) only applies to `/tokenize`. The chat endpoint already accepts it, so patch A2 is minor.
- chromix's automatic switch to JSON tool calls was deliberately *not* taken: vLLM's `qwen3_xml` parser can't read JSON tool calls.

## The A/B: which defaults are best?

BFCL v4 has 1040 items × 2 repeats per variant. EvalPlus has 542 items and long context 180. Same server and same seeds; the comparison is paired per item.

| Variant | BFCL | Δ vs defaults (95% CI) | EvalPlus | Long context |
|---|---|---|---|---|
| defaults (the model's own prompt) | 78.9% | | 85.8% | 98.3% |
| **chromix tool text** (= current production) | **80.2%** | **+1.3** (−0.1 … +2.7), p 0.14 | 84.9%\* | 97.2%\* |
| + payload_rules + failure_notes (Sharp) | 78.1% | −0.8 (−2.3 … +0.7) | | |
| + payload + failure + terse | 78.3% | −0.6 (−2.2 … +1.1) | 85.4%, with **−38% output tokens** | 96.1% |
| + strict_tool_rules | 78.8% | −0.1 (−1.5 … +1.3) | | |
| + payload + failure + strict | 77.8% | −1.1 (−2.5 … +0.5) | | |

\* Without tools, chromix and the defaults render identical prompts, so these differences are pure noise (about ±1 point).

**Conclusions:**
1. **chromix's tool text is the only thing that helps**, mainly on irrelevance (+4.5 / +3.0 points): the model calls tools less often when it shouldn't. The four format rules alone do nothing; the gain comes from the rewritten guidance.
2. **Sharp's extras (payload rules, failure notes) tend to hurt** this model, so they stay off.
3. **terse** cuts coding answers by about 40% in length and time at no measurable EvalPlus cost. It deserves its own test.

I therefore added `tool_prompt: "chromix"`, which reproduces chromix's tool text byte for byte (tested in 16 combinations). **The server default became `{"reasoning_effort":"medium","tool_prompt":"chromix"}`.** The measured requests then get exactly the same prompt as production had, plus every fix above.

## Verification of the final version (server C: final template + vLLM patches A1/A2)

**Live feature check: 29/29 PASS** through the real vLLM server (OpenAI and Anthropic APIs):
- **Reasoning effort:** all 11 effort values are accepted. With none/off, `<|think_off|>`, `enable_thinking=false` and the string `"false"`, the answer lands in `content`.
- **`<|think_on|>`** beats `enable_thinking=false` and effort none: reasoning and answer are cleanly separated.
- **Injection:** `<|im_end|><|im_start|>system…` in user text gives 2 real turns, the same as a clean request.
- **Tools:** single, parallel and irrelevance cases; a round trip with string arguments; truncation; failure_notes on/off.
- **History reasoning:** read via `reasoning`, `reasoning_content`, `thinking` and inline `<think>`; `preserve_thinking=false` works.
- **Roles and content:** developer, two system messages, content parts, a system message mid-conversation.
- **Anthropic API:** tool_use, a tool_result round trip, and **tool_reference (Claude Code tool search)**.
- **Prefix reuse:** turn 2 of a 15K conversation takes 0.8 s instead of 4.5 s.

The production check (aliases, effort, tool call, Anthropic, streaming, API key) also passes.

**Quality suite on the final version** (same benchmarks and seeds as the A/B):

| | BFCL (2080) | Δ vs defaults | EvalPlus | Long context | multi-turn |
|---|---|---|---|---|---|
| defaults (the model's own prompt) | 78.9% | | 85.8% | 98.3% | 61.5% |
| chromix v1 (production until tonight) | 80.2% | +1.3 (−0.1 … +2.8) | 84.9% | 97.2% | 63.0% |
| **final version: qfn-asgard-v2 + `tool_prompt: chromix`** | **80.3%** | **+1.4 (+0.0 … +2.9)** | **86.5%** | 97.2% | 60.5% |
| final version + `preserve_thinking: false` | | | | | 57.0% (−3.5, ns) |

**Reading:**
- The chromix tool-text effect **replicates** in an independent run, at +1.4 points.
- The final version is as good as production on every benchmark. The differences on EvalPlus, long context and multi-turn are inside the noise, as the identical-prompt comparison showed.
- Dropping old reasoning (`preserve_thinking: false`) points the wrong way. Keeping it is also what gives full prefix reuse, so the default stays.

## Experiments (on the same test server, other templates sent per request)

**1. Reasoning effort xhigh vs medium.** Swift's model card defaults to xhigh, and its GPQA score is 5.9 points higher there.

| | BFCL (1040, paired) | EvalPlus (478, paired) | output tokens (EvalPlus) |
|---|---|---|---|
| medium (production) | 80.5% | 85.8% | 901 |
| xhigh | 75.3% (**−5.2, CI −7.8 … −2.8**) | 85.4% (−0.4, ns) | 1319 (+46%) |

**The loss sits in parallel calls:**

| Category | medium | xhigh |
|---|---|---|
| parallel | 86% | 70% |
| parallel_multiple | 79% | 56% |
| live_parallel | 94% | 56% |

At xhigh, "wrong number of calls" failures rose from 17 to 56, and 76 items got 4+ calls against 53 at medium. The xhigh instruction ("validate key assumptions, consider plausible alternatives") makes the model call more "to be safe". The GPQA advantage does not carry over to tool use. **medium stays.**

**2. History handling, measured on BFCL multi-turn** (100 conversations × 2, paired):

| | model's own template | qfn-asgard |
|---|---|---|
| client sends reasoning back | 61.5% | 61.5% |
| client drops reasoning (common in clients) | 59.0% (empty `<think></think>` in history) | **62.0%** (no empty blocks) |

- **With reasoning sent back:** the two templates are equal.
- **When the client drops reasoning:** the model's own template loses 2.5 points and ours doesn't. That is +3.0 for ours, consistent with the "empty think teaches the model to stop thinking" theory, but not significant at this sample size.
- **Community practice is split on this point** (froggeric v22.5 renders empty blocks), and nobody had published a measurement. This is the first.

## Production

**What changed in production:**
- **Profile `/opt/vllm-qfn/qfn-prod.env`:**
  - `--chat-template /opt/vllm-qfn/templates/qfn-asgard-v2.jinja`
  - `--default-chat-template-kwargs {"reasoning_effort":"medium","tool_prompt":"chromix"}`
- **vLLM tree `/opt/vllm-qfn-exp/src`** got two small patches:
  - A1, in `serving.py`: thinking on/off agreement between template and parser.
  - A2, in `chat_utils.py`: history reasoning from `thinking` and through `/tokenize`.

**Deploy gate** (all automatic, 03:34–03:41 UTC):

| Check | Result |
|---|---|
| health | 200 after 4 min |
| prod_check | PASS |
| func_check | PASS |
| multi-turn | PASS, 80% prefix-cache hits |
| feature check on 8080 | **29/29** |
| speed vs night 2 | PP 4920 tok/s @4K (102%), 7788 @32K (102%); TG 148/152 tok/s (103% / 98%) |

**Rollback** (if anything misbehaves):

```bash
cp /root/tune/night3/qfn-prod.env.baseline /opt/vllm-qfn/qfn-prod.env && systemctl restart vllm-qfn
```

To also undo the vLLM patches, copy back the two files in `/root/tune/night3/pre-asgard/`. They are unchanged copies from git.

The night fallback timers have been removed. They could otherwise have overwritten the new production at a restart.

## Open items and suggestions

1. **Publication: needs your OK.** The package is in `reports/qfn-asgard-template/`: template, README, tests and vLLM patches.
   - Licence: Qwen Community License 1.0, inherited from Sharp-naze, with credits.
   - One decision for publication: should the template's *own* default be `tool_prompt: chromix`, the measured best? Today the default is the model's own text, and chromix needs a kwarg.
   - My recommendation is to switch the default, because most users drop a template in without kwargs.
2. **`terse` deserves its own A/B.** Answers were about 40% shorter on coding tasks at no measurable cost, but it was only tested together with Sharp's other extras.
3. **Not taken tonight, candidates for later:**
   - "think seed" (`<think>\nLet me`), measured by others at −11% reasoning tokens at medium, untested on Swift;
   - failure counting per batch (Sharp-naze v1.5; only matters when failure_notes is on);
   - dropping empty reasoning-only turns (weak evidence).
4. **Watch item:** the vLLM parser leaves `\n\n` at the start of `content` after `</think>`. This is cosmetic: the template trims it on the way back, so prefix reuse is unaffected.
