---
title: "Code Review in the Age of Agents"
date: 2026-03-04T09:00:00-05:00
draft: false
tags: ["code-review", "claude-code", "ai-coding", "opinion"]
params:
  author: "Jim Robinson-Bohnslav"
---

In World War I, new technology like the machine gun, barbed wire, and artillery suddenly shifted the balance from offense to defense. Cavalry charges were out — trenches and machine guns were in. The same epochal shift is happening right now in software engineering, but on the side of offense. AI coding agents (Claude Code, Codex, Cursor) have made *writing* code dramatically faster. But *reviewing* code -- defense -- is still happening at human speed. The bottleneck has flipped.

{{< figure src="/images/code-review-castle.jpg" alt="AI robots storming the old guard code review fortress" caption="Photo credit: Nano Banana" >}}

Even on our small team, our PR backlog is growing. Our outdated requirement and culture is that every PR needs an approved owner to sign off on the changes. This is slowing down our entire cycle and making us all work on bloated branches that we have to check out to perform our daily work. Something's gotta give.

### ML comes for the software engineers

This isn't the first time we've seen this pattern. When machine learning started eating computer vision, the actual algorithm —  hand-crafted morphological operations, SIFT features, and rule-based systems — stopped mattering. What mattered was your **inputs** (good data) and your **outputs** (clean labels). What happens in the middle is up to the neural network.

As Karpathy put it: *gradient descent can write better code than you.* At the time, he meant feature extractors for computer vision. Now it means literal software.

{{< x user="karpathy" id="893576281375219712" >}}

The lesson for code review is the same: **stop inspecting the algorithm. Verify the inputs and outputs.** The "inputs" to an agentic coding process are requirements, designs, constraints, and high level decisions; the "outputs" are working artifacts: features, metrics, screenshots, data products, etc.

## What PR Authors Should Actually Do

- **Put your inputs in the description.** This is core to understanding the agent's implementation: what did you tell Claude or Codex to *do*? Did you prompt it to prioritize speed or reliability? What requirements or constraints did you set?
- **Put your outputs in the description.**  You built a web app? Put a screenshot in. How does it look? How did it change? Is this complex feature something only the engineer wants, or does it benefit the whole team?

## What Reviewers Should Actually Do

### Verify Inputs — Is the Design Sound?

- **Architecture:** Did the PR author couple components that should be separated? Will adding a feature on the roadmap require a rewrite? Basically, classic system design.
- **Blast radius:** This is the big one. Is this PR going to break other team members' ongoing work? Have you checked with them? Does the code change *risk *=potentially* breaking others' work? Did the authors actually run the actual prod code paths — not just unit tests — and verified nothing downstream is affected?
- **Technology choices vs. team context:** Sure, the authors vibe-coded an internal tool that uses Rust, but nobody on the team knows Rust. Claude Code will go down at some point. It's also not *quite* as intelligent as a human SWE: we may have to dig in and fix bugs ourselves from time to time. *Someone* on the team needs a mental model of the code.

Blast radius is the irreducibly human part of review. No AI reviewer knows your team, your org, or who's mid-flight on what. Review agents don't know the team has a big upcoming deadline, so modules A, B, and C absolutely can't break until next Tuesday.

### Verify Outputs — Does It Actually Work?

AI review tools (e.g. Cursor Bugbot, or prompted Claude Code + Codex) will do a more careful line-by-line inspection than any busy engineer ever will. There's too much code to pore over, and the agents are more thorough at that level anyway.

Instead, reviewers should focus on **proof that the code works as intended:**

- The PR rewrote a data pipeline? Did the authors run it on prod data? How fast was it? Are the outputs identical? If they aren't, did the authors communicate with downstream consumers?
- The PR changed the training pipeline? Did the authors do an overfitting test? Is it faster, slower, better, worse?

The reviewer becomes a verifier, not a code inspector.

## Do's and Don'ts for reviewers

### Don't

- **Don't just throw a PR at Claude and say "review this."** The PR submitter has (hopefully) already done that. You'll get the same generic review they already addressed.
- **Slap LGTM with no verification**: check the PR description. Think for a minute with your human brain. What do you actually want to know about how (and whether) this code works?

### Do

- **Manually review known danger zones**. I wrote code in our training pipeline that handles model parallelization (FSDP2), activation checkpointing, torch.compile, custom kernel integration, etc. These components interact in ways that it's difficult to build a mental model of. I absolutely don't believe Opus 4.6 "understands" these interactions (maybe Codex 5.3-xhigh does). If someone vibe-coded a rewrite, I'd throw it out with prejudice. You, the human, should know which areas are chaotic — small changes lead to large outcomes — and which aren't.
- **Build your own opinionated review prompt.** Again, no one has time for line-by-line review, but you can take 30 minutes and distill your reviewing style into a good prompt. Take what *you* specifically look at and care about: your style, your pet peeves, your architectural instincts, and encode that as a personal skill or prompt. Run *that* against the PR. This way the team gets the diversity of perspective that makes code review actually valuable, not N copies of the same generic AI review.
- **Make a "deslop" or style skill and run it on your own branch periodically.** Clean up AI-generated code *before* it hits review. Don't make your reviewer clean up after your agent.

## The Bitter Pill

Unfortunately for those of us who are used to strict reviews upholding code quality, we must *learn to let go.* I find myself thinking: I'm going to get to review that 10,000 line PR tomorrow morning. But then I don't, because I'm working on my own 10,000 line PR. I'm never going to get to it — and my engineers aren't either. If the outputs work, the architecture is sound, the blast radius is checked, the tests exist, and your personalized review turns up nothing egregious — **slap a LGTM on it.**

The codebase as a whole needs to move at the speed that coding agents make possible. Don't let the humans — or your outdated process — be the bottleneck.
