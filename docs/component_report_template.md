# How we document a component / experiment — read this before writing one

**If you are an AI assistant picking up this project in a new chat and the user asks you to "write up" or "document" a component/idea that's been run (or has failed, or is finished either way): follow this file exactly. Don't improvise a different structure.** A worked example, filled out for real, is [`results/component1_exact_classifier.md`](../results/component1_exact_classifier.md) — read it alongside this file if anything below is unclear.

## The idea behind this convention

Every idea in this project starts as a plan (in `planning/`), gets built, gets run, and produces a result — good, bad, or a dead end. By the end of the project, the goal is one exhaustive, honest list of every experiment tried, in one consistent format, so nothing gets lost and nothing gets quietly forgotten just because it didn't work. **A failed or dead-end idea gets exactly the same quality of write-up as a successful one** — a dead end that's clearly documented saves someone from trying the same thing again; a dead end that's never written up doesn't.

**The pairing this project already uses, made explicit:** an idea lives in `planning/NN-*.md` before it's run. Once it's actually been run (fully, partially, or abandoned partway through), it gets a matching report in `results/`. The planning doc is the *intent*; the results doc is the *outcome*. Never edit a planning doc to pretend it predicted the outcome — the outcome goes in its own file, in `results/`.

## The one rule that overrides everything else here

**Never write up an experiment, a result, or a number that wasn't actually run and checked.** If something is planned but not yet executed, it does not get a report yet — it stays in `planning/`. This project has one existing standing rule about this already (see the project's own memory: "don't overclaim experiments") and this template exists partly to make that rule impossible to accidentally violate. Every section below that asks for results should be left honestly blank, or marked "not yet run," rather than filled with a guess.

---

## The required sections, in this order

### 1. Title and status
One line: what this is, plus a status marker. Use exactly one of:
- ✅ **Done — result confirmed** (ran, checked, numbers are real)
- ⚠️ **Done — partial / mixed result** (ran, but incomplete, or the result cuts both ways)
- ❌ **Dead end** (ran, and it didn't work — say so plainly, don't soften it)
- ⏸️ **Abandoned before completion** (started, stopped for a stated reason before getting a result)

### 2. One-line summary
A single sentence a busy reader could get the whole point from, without reading further.

### 3. Origin
Which `planning/NN-*.md` file this idea came from, and which specific component/idea number within it. If it didn't come from an existing plan doc (a spontaneous idea, tried quickly), say that instead — don't force a fake origin.

### 4. The issue this targets
Two cases:
- **It targets a specific failure hypothesis already established in this project.** Name it precisely — which phase found it, what the actual number was (e.g. "Phase D found real, consistent negative BWT of −0.68 to −4.68 across every Office-Home ordering"). Don't paraphrase vaguely; quote the finding.
- **It doesn't map to an existing failure hypothesis.** Say so directly, then state plainly what new problem or question is actually being addressed, and why it's worth addressing at all. Don't force-fit it into an existing hypothesis just to have something to point at.

### 5. Why we tried this approach specifically
The reasoning that led to this particular method, not just the problem it addresses. What made this the thing worth trying, out of the possible options?

### 6. Method
The real, detailed, technical description — equations, exact mechanisms, exact design decisions, and *why* each decision was made (especially any deliberate deviation from an original/ideal approach, e.g. a substitution made to keep something computationally tractable — state what was substituted and why, every time). This section is for the permanent record, not a presentation — don't simplify away detail that a future reader (or reviewer) would need to actually evaluate the method.

### 7. Dataset(s) used, and why
Which data, and the specific reason it was chosen for *this* experiment — not a generic "it's a standard benchmark," but why it's the right test for the specific question in §4. If more than one dataset was used, say what each one specifically confirms or rules out that the others don't.

### 8. Code
Every file actually involved, with real repo-relative paths — the implementation, the experiment/validation script, anything else load-bearing. A future reader should be able to go straight to the code from this list, with no searching.

### 9. Results
The actual numbers. Tables where there's more than a couple of values. Cite the results file directly if there's a `results/*.json` backing it (link it). No result belongs in this section unless it was actually produced by a real run — per the one rule above.

**Include a graph wherever one would make the result easier to read at a glance than the table alone** — comparisons across more than ~3 conditions, anything with a trend across an axis (probe size, epoch, domain ordering), or anything you'd want to point at on a slide. These reports get shown to the project's advisor, not just kept as an internal record — a table of numbers makes them re-derive the shape of the result themselves; a graph shows it directly. Generate figures with a small `results/generate_<report-name>_figures.py` script (matplotlib, following the existing `generate_phase_*_figures.py` scripts as the pattern), save PNGs to `results/figures/`, and embed with `![caption](figures/<name>.png)`. A figure is exempt from the "never write up a result that wasn't actually run" rule only insofar as it's a *plot of numbers already in §9* — it must not visualize anything not already backed by a real `results/*.json`. Skip the figure only when the result is genuinely a single number or two — don't chart for its own sake.

### 10. What this means
Plain-language interpretation of the numbers in §9 — not a restatement of them, an explanation of what they actually tell us. This is where nuance goes: if a result is real but narrower than it first looks, say exactly how it's narrower (this project has a strong track record of doing this honestly — Phase A's null result, Phase F2's reframed finding, Phase E2's "target isn't depressed" nuance are the models to follow).

### 11. Verdict
Did it solve the issue from §4? Completely, partially, or not at all? If there's a literature check relevant to novelty (was this already published elsewhere), state it here too, plainly — see `docs/new_methodology_report.md` §6 for the standard this project already holds itself to.

### 12. What's next
**The 90% bar, apply it before writing anything here:** only list a next step if you're at least 90% confident it will produce real, non-redundant information — genuinely uncertain in outcome, not something already answerable from evidence this project has already collected — and you can state plainly *why* you're that confident (what specific evidence, already in hand, makes you sure). If nothing clears that bar, say so directly: *"no further step under this component is confidently warranted right now."* Don't manufacture a plausible-sounding step just to fill this section — a padded next-steps list is exactly how a project drifts into a chronic loop of low-value follow-up experiments that each sound reasonable in isolation but never converge on anything. An honest "nothing else is worth running" is a complete, valid answer, not a gap in the report.

Two cases, and be honest about which one applies:
- **There's a real next step that clears the 90% bar.** Say exactly what it is, why it follows from this result, and name the specific reason you're confident it'll produce something new (not just plausible-sounding — if the likely outcome is already predictable from data this project already has, it doesn't clear the bar, even if it would still technically confirm something).
- **This is a dead end, or nothing else clears the bar.** Say so in plain words — "this approach doesn't work, here's why, we're not pursuing it further" or "no further step here is confidently worth running" — and note what (if anything, itself meeting the same bar) should be tried instead to address the original issue from §4, elsewhere in the project if not here. A dead end is not a failure of the project; a dead end that isn't written down honestly is. Neither is "no more next steps" — an unpadded stop is the honest outcome sometimes, not a shortcoming of the report.

---

## Where these files live

- Plans (before running): `planning/NN-short-name.md`
- Reports (after running, any outcome): `results/componentN_short-name.md` (or `phaseX_short-name.md` for anything that's testing the original diagnosis rather than the new methodology — keep using whichever prefix matches which side of the project it belongs to)
- This convention document itself: `docs/component_report_template.md` (this file) — update it if the convention itself needs to change, but don't fork a second version of it
