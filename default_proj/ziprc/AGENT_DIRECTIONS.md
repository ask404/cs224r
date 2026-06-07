# From ZIP-RC-Lite to Agent Systems — Research Directions

How the introspection head generalizes to the problems facing real LLM-agent systems, and
the experiments that would test it. Grounded in recent failure-mode, SLM, and cost-aware-eval
literature plus agent-harness practice.

## The thesis
**A small model that predicts its own *reward and cost* in the same forward pass is the
missing control signal for cost-aware, failure-aware agent systems** — and it's *most*
valuable in SLM-first stacks, where a separate verifier/router would blow the very cost
budget CLEAR shows matters. Our results already supply the prerequisites:
- calibration matches the paper (TV 0.52; `PROGRESS.md`);
- success is predictable from the **first token** (AUC 0.85; `value_first`);
- the signal is **policy-agnostic** (`CROSS_POLICY.md`) — a portable, not per-model, probe.

## The wedge: prospective, not post-hoc
Failure-analysis work (Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, NeurIPS 2025;
Zhu et al., *Where LLM Agents Fail and How They Can Learn From Failures*,
[2509.25370](https://arxiv.org/abs/2509.25370)) diagnoses failures **post-hoc**, after the
trace completes. **No one predicts the cascade *before* it happens, at zero overhead.** ZIP-RC
does exactly that — this is the opening.

---

## Direction B (flagship) — Prospective failure prediction & early intervention
**Problem:** agents fail in taxonomizable ways and failures *cascade*, burning compute on
doomed trajectories.
**Idea:** a ZIP-RC head predicts, *at each agent step*, `P(trajectory fails)` and
`E[remaining cost]`; below threshold ⇒ **abort / replan / escalate early**. Train it on
labeled failure traces using Cemri's 14 modes / Zhu's 5 dimensions as a **structured
verifier** — *exactly our 3-outcome verifier generalized to an agent failure taxonomy* ⇒
predict *which* failure mode ⇒ targeted recovery.
**Why it matters:** closes the loop AgentDebug leaves open (prospective prediction → their
post-hoc recovery); attacks CLEAR's Cost + Reliability axes.
**First experiment:** instrument ALFWorld / GAIA / WebShop (Zhu's benchmarks); predict
per-step trajectory success + mode; compare *early-abort-and-replan* vs *run-to-completion*
on compute-saved and final success. **The most direct extension of this project.**

## Direction A — Introspective escalation for SLM-first stacks
**Problem (Belcak et al., *SLMs for Agentic Systems*, NVIDIA 2025; Bhatia et al., *CLEAR*,
[2511.14136](https://arxiv.org/abs/2511.14136)):** SLMs are 10–30× cheaper but fail more;
*when* should an SLM answer vs escalate to a frontier model? Accuracy-only agents are
**4.4–10.8× more expensive**; a separate router is itself a cost.
**Idea:** the SLM predicts `P(success)` + `E[cost]` in the *same forward pass* and escalates
only when expected utility favors the big model — **zero marginal cost for routing**. Our
**cross-policy transfer** result is the enabler (the signal isn't tied to one model).
**First experiment:** SLM+LLM cascade; plot the CLEAR cost–efficacy Pareto for *introspective
routing* vs always-SLM / always-LLM / separate-router / logprob-threshold.

## Direction C — Introspection-driven harness meta-control
**Problem:** harnesses (Stanford [meta-harness](https://github.com/stanford-iris-lab/meta-harness),
Claude-Code-style loops) set retries, per-step sample budget, context-compaction timing,
subagent spawning, and tool selection by **heuristics or offline search** — never a *live*
self-signal. Meta-harness optimizes scaffolding around a *frozen* model; its control surfaces
(*escalation hints, context construction, tool presentation*) are exactly where a signal plugs in.
**Idea:** generalize our token-level adaptive sampler to **harness-level meta-actions** — the
model's `(reward, cost)` prediction becomes the live input to "how many samples / retry? /
compact now? / spawn a subagent?". A drop-in **metacognitive layer**.
**First experiment:** A/B the introspective controller vs default heuristics on a task suite;
measure all five CLEAR axes.

## Direction D — Calibrated selective agents (enterprise Assurance/Reliability)
**Problem (CLEAR):** enterprise agents must *know when to abstain / escalate to a human*
rather than confidently fail; confidence is poorly calibrated or expensive.
**Idea:** ZIP-RC reward prediction as a zero-overhead **calibrated abstention signal** → a
selective agent with a tunable risk–coverage curve. Our finding that the *reward* half
calibrates well (TV 0.52) but the *length* half doesn't is the precise research question.
**First experiment:** risk–coverage curve (abstain when `P(success) < τ`) vs logprob-confidence
and separate-verifier baselines; report accuracy-on-answered, coverage, ECE, and cost.

---

## Recommendation + the one de-risking test
Pursue **B as flagship, with A as companion** (B = "when to give up," A = "when to escalate" —
same signal, two meta-actions). Both are grounded in 3 of the 4 papers and extend our
structured-verifier + introspection head most directly.

**Riskiest assumption — test first, cheaply:** *does zero-overhead success prediction work on
open-ended agent tasks the way it does on verifiable ones?* ~1-day keystone: train a ZIP-RC
head on a verifiable tool-use/coding agent's traces and measure **prospective
failure-prediction AUC at step 1, 2, 3…**. If the `value_first`-analog clears ~0.7 early (like
our 0.85 on Countdown), the agenda is viable; if ~chance, the signal needs richer supervision
before any harness integration. *(Same "gate the expensive build behind a near-free test"
discipline that caught the tied-embedding bug.)*

## Honest caveats
1. **Agent rewards are often unverifiable** — you'd lean on an LLM-judge verifier (cost +
   noise, like our Haiku judge).
2. **The cost/length head was our weakest component**, and cost-prediction is half the agent
   value prop — harden it first.
3. **Multi-agent credit assignment** (which agent's step doomed the run?) is genuinely hard
   (Cemri flags it) and is required before Direction B extends cleanly to multi-agent.
4. **Cross-*base* transfer is unproven** (`CROSS_POLICY.md`): our transfer result is within one
   base model — a portable probe across model families is a separate, harder question.

## References
- Cemri et al., *Why Do Multi-Agent LLM Systems Fail?*, NeurIPS 2025.
- Zhu et al., *Where LLM Agents Fail and How They Can Learn From Failures* — https://arxiv.org/abs/2509.25370
- Belcak et al., *Small Language Models for Agentic Systems: A Survey*, NVIDIA, 2025.
- Bhatia et al., *Beyond Accuracy* (CLEAR) — https://arxiv.org/abs/2511.14136
- Manvi et al., ZIP-RC — https://arxiv.org/abs/2512.01457 ; predecessor — https://arxiv.org/abs/2410.02725
- Stanford meta-harness — https://github.com/stanford-iris-lab/meta-harness
