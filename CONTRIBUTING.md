# Contributing

Thanks for helping maintain this list. The goal is a collection a researcher can
*trust*: every claim traceable to a primary source, every venue label verified, and
every summary written to inform rather than to sell. This document explains the
standard so that future updates are reproducible by anyone.

- [What belongs here](#what-belongs-here)
- [What does not](#what-does-not)
- [Primary sources](#primary-sources)
- [Dates and chronology](#dates-and-chronology)
- [Venue status](#venue-status)
- [Entry format](#entry-format)
- [Writing the summary](#writing-the-summary)
- [Code links](#code-links)
- [Categories](#categories)
- [Keeping "Recent Additions" bounded](#keeping-recent-additions-bounded)
- [Corrections and removals](#corrections-and-removals)
- [Pull request checklist](#pull-request-checklist)

## What belongs here

Add a paper when at least one of the following holds.

1. Muon is the main algorithm or the main subject of analysis.
2. The work proposes a direct Muon variant, or modifies one of its components:
   orthogonalization, matrix-sign computation, spectral transformation,
   normalization, scaling, momentum, adaptivity, second-moment estimation,
   regularization, distributed execution, quantization, or memory layout.
3. The work gives a substantive theoretical result about Muon specifically — a
   convergence analysis, a lower bound, a counterexample, a negative result, or a
   characterization of a limitation.
4. Muon is a central experimental variable in an application study, with meaningful
   tuning, ablation, or comparison — not merely one baseline among many.
5. The work is historically necessary to explain Muon's intellectual lineage:
   norm-constrained steepest descent, matrix-sign updates, polar factors, or closely
   connected spectral optimization formulations.
6. It is an official implementation or official framework integration that materially
   affects practical use of Muon.

**Negative and mixed results are as welcome as positive ones.** A paper reporting
that Muon failed to reproduce an advantage, or that a competing explanation fits the
data better, is more valuable to this list than another incremental variant. Please do
submit them.

## What does not

Exclude, or place in the clearly labelled **Adjacent Matrix and Spectral Optimizers**
section, when:

- Muon appears only once in a related-work paragraph;
- Muon is one baseline among many with no meaningful analysis;
- the connection rests only on shared terminology;
- there is no accessible primary source;
- the entry duplicates one already present;
- the work is commentary or a summary without original technical content;
- you cannot explain the relevance in one precise sentence.

Do not add papers to raise the count. A shorter list that a reader can trust is worth
more than a longer one they have to re-verify.

## Primary sources

Every entry needs a primary-source link. Use sources in this order of preference:

1. Official conference or journal pages — OpenReview forums, PMLR, journal sites,
   official proceedings.
2. Canonical arXiv abstract pages (`https://arxiv.org/abs/<ID>`, **without** a version
   suffix) or arXiv API metadata.
3. Official project or author repositories linked from the paper.
4. Official framework documentation — PyTorch, DeepSpeed, NVIDIA NeMo or Megatron, or
   another first-party source.
5. Author-written technical posts, only where they are historically relevant or contain
   implementation details unavailable in a paper.

Prefer an abstract or venue landing page over a direct PDF link, and prefer a canonical
OpenReview **forum** URL over a bare `openreview.net/pdf/<hash>` link — hash URLs break
when a paper is revised.

**Do not** use as final evidence for a title, date, claim, or publication status:
search-result snippets, Semantic Scholar or Papers With Code summaries, third-party
blog posts, reposted abstracts, SEO pages, automatically generated paper summaries, or
other awesome lists. Other curated lists are fine as *discovery* aids — use them to
find candidates, then verify each candidate independently.

## Dates and chronology

**Order by the first public version.** For an arXiv entry that is the `[v1]` date in
the submission history, not the latest revision date. Using a revision date distorts
the chronology, sometimes by more than a year: several papers in this list have been
revised three or four times since their first release.

If an arXiv abstract page will not render its metadata, the DataCite record for the
paper's DOI (`https://api.datacite.org/dois/10.48550/arXiv.<ID>`) carries per-version
`Submitted` timestamps and the arXiv comment field, and is an acceptable fallback.

Note that an arXiv identifier's month can differ from the submission month if the paper
was held before announcement. Record the submission date and, if the discrepancy is
likely to confuse, mention it.

If you can only establish the month, say so rather than guessing a day.

## Venue status

**Label a paper as published only when an official venue page or the arXiv `Comments`
field says so.** Not the paper's LaTeX template — several preprints in this area are
typeset with an ICML or ICLR style file and were never accepted anywhere. Not a
submission header, which says "Submitted to", not "Published as". Not another list's
claim.

Acceptable evidence: an OpenReview forum showing a decision; a PMLR or official
proceedings entry; a conference virtual-site poster or oral page; an arXiv `Comments`
field explicitly stating acceptance.

If you cannot verify it, write `Preprint`. That is not a demotion — it is an accurate
statement about what is currently checkable. Be specific where a venue has parts: a
workshop paper is a workshop paper, and naming the workshop is better than writing
"ICML WS".

If a paper's title or author list changed between versions, use the **current** ones
and note the change in the entry when a reader might otherwise search for the old title.

## Entry format

```markdown
- **[Exact paper title](canonical primary-source URL)** — First Author, Second Author,
  Third Author, et al., Venue Year or Preprint, Year. One concise sentence explaining
  the contribution and its scope. [Code](official repository)
```

Conventions:

- Use the exact title, with its own capitalization.
- List up to three authors, then `et al.`
- Write `Preprint` where the venue is unverified; write the verified venue otherwise,
  including the distinction (Poster, Spotlight, Oral, workshop) when it is confirmed.
- The trailing year is the year of first public release. Omit it when it is the
  same as the venue year (`NeurIPS 2025.`); include it when they differ
  (`ICLR 2026, 2025.`), since the gap between release and publication is itself
  useful information.
- Code links are optional; include one only if it is first-party.
- Within a section, sort newest first by first public date.

Spelling and capitalization used throughout: **Muon**, **AdamW**, **Newton–Schulz**
(en dash), **FSDP**, **ZeRO**, **LoRA**, **muP**.

Use mathematically accurate terminology: *polar factor*, *semi-orthogonal*, *matrix
sign*, *spectral norm*, *singular values*, *matrix-valued parameters*. Do not describe
a rectangular polar factor as "orthogonal" without qualification.

## Writing the summary

One sentence. In your own words. Never paste or lightly paraphrase an abstract.

A good summary says what the paper *does* and what its *scope* is. Scope is the part
most often omitted and the part readers most need — model size, task, whether the
result is a theorem or an observation, and what the reported number actually measures.

Distinguish carefully:

- a theorem from an empirical observation;
- a mechanism the authors *propose* from one the evidence *establishes*;
- convergence in a mathematical setting from performance in neural network training;
- step efficiency from token efficiency from wall-clock efficiency;
- single-device from distributed results;
- pretraining from fine-tuning;
- small controlled experiments from large-scale ones;
- optimizer effects from changes in parameterization, initialization, learning rate,
  weight decay, batch size, or training budget.

Useful phrasings: "the authors argue that…", "the paper reports … in the evaluated
setting", "the result is established under … assumptions", "evidence is mixed
across…", "this does not by itself establish…", "under matched tuning…", "the reported
advantage is measured in steps rather than wall-clock time".

Avoid: "Muon always outperforms AdamW", "solves optimization", "orthogonalization is
definitively the reason", "proves that Muon is better for LLMs", and "state of the
art" without a precisely verified benchmark.

**Quantitative claims need context.** Not "2× faster" but "reports matching AdamW's
loss at roughly 52% of the training FLOPs in the authors' scaling-law setting". Not
"8% faster" but "reports an 8% throughput gain in one distributed configuration".
Include a number only when it is verified from the paper and helps a reader understand
the contribution — and always say what unit it is in.

**No rankings.** Do not rank entries by prestige, citation count, author reputation, or
social-media attention, and do not add "top N" sections. The reading paths in the
README are grouped by *purpose*, and each item carries a reason it belongs there.

**No preferential treatment.** The same inclusion and prominence standards apply to
every author, including the repository owner and existing contributors.

## Code links

Link code only when it is **first-party** — the authors' own repository, or the
official organization repository the paper points to. Do not link third-party
reimplementations, and never describe an unofficial wrapper as official.

Anonymous review links (for example `anonymous.4open.science`) are not permanent; you
may mention that code exists behind one, but do not present it as a repository link.

For the **Implementations and Ecosystem** section, state where Muon is implemented,
which parameter groups are eligible and what handles the rest, what distributed support
actually exists (be specific: DDP, FSDP2, ZeRO stage, tensor-parallel, or none), any
implementation-specific behaviour that affects results, and the date you checked. If
you could not confirm something, say "unknown" rather than assuming.

## Categories

Each paper lives in **exactly one** topical section. If a paper spans several — and
many do — pick the one matching its primary contribution, and cross-reference it from
the narrative in `docs/research-landscape.md` if that helps. Do not list the same paper
twice; duplicate entries drift apart as one is updated and the other is not.

If your paper genuinely does not fit any existing section, say so in the pull request
rather than forcing it into the nearest one. Adding or merging a category is a
reasonable outcome.

If you are unsure whether a paper is relevant, **say so in the pull request**. An
explicit "I think this qualifies under criterion 4 but the ablation is thin" is far more
useful than a confident entry that a maintainer has to re-verify from scratch.

## Keeping "Recent Additions" bounded

`Recent Additions` is a rolling window, not an archive. The policy is:

- include papers first publicly released in roughly the trailing ten weeks;
- cap the section at **15 entries**;
- when adding an entry, remove the oldest ones that fall outside the window;
- every entry there must **already** have a permanent entry in its topical section —
  `Recent Additions` is a pointer, not a second home.

When you update the section, also update the `Last updated` and
`Research coverage through` dates at the top of the README and, if you touched it,
`docs/research-landscape.md`.

## Corrections and removals

Corrections are welcome and do not need to be large. Open an issue or a pull request
noting what is wrong and what the primary source says. Common and genuinely useful
corrections: a preprint that has since been accepted somewhere, a retitled paper, an
expanded author list, a link that now redirects, a version whose experimental scale
changed materially.

**Prefer reclassification and metadata correction over deletion.** When an entry is
removed, state the reason in the pull request, using one of:

- duplicate;
- broken or superseded link;
- outside scope;
- could not verify;
- commentary rather than research;
- merged into another canonical entry.

## Pull request checklist

Copy this into your pull request description.

```markdown
- [ ] I verified the title and authors against a primary source.
- [ ] I verified the venue status against an official venue page (or wrote "Preprint").
- [ ] I used the first public submission date for chronology, not the latest revision.
- [ ] I checked for an existing entry with the same arXiv ID or DOI.
- [ ] My summary is original and describes scope or limitations.
- [ ] The code link is first-party (or omitted).
- [ ] The entry is in exactly one primary category.
- [ ] I did not use promotional or universal claims.
- [ ] If I added to Recent Additions, the paper also has a permanent topical entry and
      the section is still within its cap.
- [ ] I flagged anything I was unsure about.
```
