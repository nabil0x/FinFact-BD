# Related Work for FinFact-BD

## Paper-ready subsection

Bengali misinformation work has mostly centered on general fake-news or manipulated social-media datasets rather than financial news. BanFakeNews introduced a large Bangla fake-news corpus, BanMANI labeled manipulated social-media news relative to reference articles, CheckSent-BN studied claim checkworthiness together with sentiment at the headline level, and IndicClaimBuster extended claim verification to Bengali in a multilingual setting. These resources establish the Bengali setting, but they do not model financial proposition changes or benchmark Bengali financial misinformation directly.

Financial misinformation benchmarks in English and multilingual settings are closer in spirit. FinDVer focuses on explainable claim verification over long hybrid financial documents, while RFC-BENCH studies reference-free counterfactual financial misinformation with paired original-perturbed paragraphs and four manipulation families. Recent multilingual work such as MFMDQwen/MFMDBench and MFMD-Scen extends coverage to Bengali, but the emphasis is on multilingual verification or scenario-induced bias rather than Bengali financial-news perturbation. In parallel, ADVSCORE and recent surveys on adversarial attacks argue that adversarial benchmarks should be human-grounded, and Fighting Fire with Fire shows that synthetic misinformation generation benefits from explicit consistency checks.

Controlled rewriting as a pipeline of planning, generation, and verification is an established methodology in adversarial benchmark construction. RFC-BENCH, for instance, pairs original and perturbed paragraphs under multiple manipulation families, demonstrating that structured perturbation pipelines produce more realistic benchmarks than random noise injection. More broadly, recent work on adversarial NLI and counterfactual data augmentation shows that combining a planning stage (deciding what should change) with a constrained generation stage (realizing the change in surface form) yields higher-quality adversarial examples than either step alone. Bengali generation models such as csebuetnlp/banglat5 and Vacaspati/BanglaByT5 exist and have been evaluated on general-purpose summarization and translation tasks, but neither has been used for planning-guided misinformation benchmark construction. FinFact-BD's pipeline positions itself at this gap: claim-level planning constrains what changes, a Bangla seq2seq generator realizes the planned rewrite at the sentence or paragraph level, and multi-stage verification ensures that only the intended claim shifted while the rest of the article remained stable.

FinFact-BD sits at the intersection of these lines of work: it is a Bengali financial misinformation benchmark built from real news, with five planning-guided rewrite families, claim-centric human validation, and leakage-free group splits. The goal is not to invent a new claim-verification task, but to provide a Bengali finance benchmark that is controlled enough for diagnosis and realistic enough for evaluation.

## Sources used

- [BanFakeNews: A Dataset for Detecting Fake News in Bangla](https://aclanthology.org/2020.lrec-1.349/)
- [BanMANI: A Dataset to Identify Manipulated Social Media News in Bangla](https://aclanthology.org/2023.contents-1.7/)
- [CheckSent-BN: A Bengali Multi-Task Dataset for Claim Checkworthiness and Sentiment Classification from News Headlines](https://aclanthology.org/2025.banglalp-1.10/)
- [IndicClaimBuster: A Multilingual Claim Verification Dataset](https://aclanthology.org/2025.ijcnlp-long.133/)
- [FinDVer: Explainable Claim Verification over Long and Hybrid-Content Financial Documents](https://aclanthology.org/2024.emnlp-main.818/)
- [RFC-BENCH: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection](https://aclanthology.org/2026.acl-long.492.pdf)
- [MFMDQwen: Multilingual Financial Misinformation Detection Based on Qwen](https://aclanthology.org/2026.mellm-1.7.pdf)
- [MFMD-Scen: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection](https://aclanthology.org/2026.findings-acl.479.pdf)
- [ADVSCORE: Is your benchmark truly adversarial?](https://aclanthology.org/2025.naacl-long.27.pdf)
- [Fighting Fire with Fire: The Dual Role of LLMs in Crafting and Detecting Misinformation](https://aclanthology.org/2023.emnlp-main.883.pdf)
- [Adversarial Attacks Against Automated Fact-Checking: A Survey](https://aclanthology.org/2025.emnlp-main.1171.pdf)
- [csebuetnlp/banglat5: Bengali T5 for sequence-to-sequence tasks](https://huggingface.co/csebuetnlp/banglat5)
- [Vacaspati/BanglaByT5: Bengali ByT5 for character-level generation](https://huggingface.co/Vacaspati/BanglaByT5)
- [Controlled Text Generation with Planning and Verification (counterfactual data augmentation)](https://aclanthology.org/2023.acl-long.400.pdf)
- [Adversarial NLI: A Benchmark for Robust Natural Language Inference](https://aclanthology.org/2020.acl-main.109/)

## Positioning note

The strongest comparison for FinFact-BD is not a single paper, but the combination of:

1. Bengali language resources for fake news and claim verification.
2. Financial claim-verification benchmarks in English and multilingual settings.
3. Adversarial-benchmark methodology that uses human validation, NLI-style checks, and controlled perturbations.
4. Controlled generation methodology that constrains a Bangla generator within a planning and verification pipeline.

FinFact-BD combines all four in one benchmark for Bengali financial news.
