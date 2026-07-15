# FinFact-BD Rewrite Rule Instructions

Use this file to define how each claim type should be rewritten before we encode the rules in the pipeline.

Edit the `Your instruction` and `Your logic` fields. Keep examples concrete. If a rule is uncertain, write `manual review`.

## Global Rules

Fill these first if you want one rule to apply to every rewrite family.

| Field | Your instruction |
|---|---|
| Minimum Bangla quality | Bangla should be coherent. |
| Should we allow slight paraphrase? | Yes, slight paraphrase is allowed. |
| Should we preserve original spelling exactly? | Yes. Preserve original spelling for unchanged spans. |
| Should we preserve source attribution exactly? | Yes. Preserve source attribution exactly unless the attribution text itself is the targeted misinformation. |
| Should we reject if headline/context conflicts? | No automatic rejection if the pipeline can update the headline or linked context consistently. If linked context is not updated, reject conflicts. |
| Should we reject quoted/opinion claims? | No automatic rejection. Quoted text can be changed when it is the selected claim, but attribution should remain coherent. |
| Maximum sentence length after rewrite | Same as the original, or as close as possible without breaking Bangla grammar. |
| Human validation priority | Prioritize coherent Bangla, clear contradiction, and context consistency. |

## Family 1: Numerical Fact

Numerical rewrites change a number, amount, percentage, count, rate, fiscal value, or currency amount.

| Field | Your instruction |
|---|---|
| Allow | Allow inflated or deflated changes in amount, percentage, count, rate, fiscal value, or currency amount. |
| Skip | Skip dates as numerical facts. Route dates, months, years, and fiscal periods to `temporal_shift`. Also skip IDs or ordinals unless the ordinal itself is financially meaningful. |
| Rewrite method | Rewrite numerical facts with a significant inflated or deflated value. Prefer deterministic exact span replacement when the target span and replacement are known. |
| Replacement rule | Replace the numerical fact with a value that contradicts the original claim. If something is numerically low, inflate it. If something is numerically high, make it much less significant. Small changes are not useful. |
| Verification rule | Verify contradiction based on numeric scale and boundary. The original and rewritten version should fall on opposite sides of a meaningful scale boundary. |
| Rejection rule | Reject if the numerical value does not contradict in terms of scaling, if the replacement is only a formatting change, or if the old value remains as the active claim value. |
| Your logic | Numerical contradiction should be coherent and significant. Slight increase/decrease or small scaling change does not add much value. |

### Example 1

| Item | Text |
|---|---|
| Original selected sentence | অর্থনৈতিক রিপোর্টার : ঢাকার মেট্রোপলিটন পুলিশ (ডিএমপি)-এর কাউন্টার টেরোরিজম ইউনিটকে ৫০ লাখ টাকার অনুদান প্রদান করেছে সিটি ব্যাংক। |
| Candidate rewrite | অর্থনৈতিক রিপোর্টার : ঢাকার মেট্রোপলিটন পুলিশ (ডিএমপি)-এর কাউন্টার টেরোরিজম ইউনিটকে ১০ লাখ টাকার অনুদান প্রদান করেছে সিটি ব্যাংক। |
| Default implementation idea | Exact span replacement: `৫০ লাখ টাকার` -> `১০ লাখ টাকার`; preserve all other tokens. |
| Why this works or fails | This fails as a strong misinformation case because `৫০ লাখ টাকা` to `১০ লাখ টাকা` does not create a large enough scale contradiction. A change such as `৫০ লাখ` to `৫ কোটি`, `৫০ কোটি`, or `১০০০ টাকা` would be more meaningfully contradictory. |
| Your final rule for this kind of case | Make changes that look significant in financial scale while keeping the sentence coherent. |

## Family 2: Entity Replacement

Entity rewrites replace a financial institution, company, regulator, ministry, market, or organization.

| Field | Your instruction |
|---|---|
| Allow | Allow entity replacement for financial institutions, companies, regulators, ministries, countries, markets, or organizations. |
| Skip | No broad skip category by default, but reject if replacement creates incoherent grammar or unresolved headline/context conflict. |
| Rewrite method | Replace the original entity with an entity that should not naturally be in that role or setting. If needed, update linked headline/context mentions consistently. |
| Replacement rule | Replace with a different-role or wrong-belonging entity, not a near-identical same-class entity. The replacement should create meaningful misinformation, not just a subtle peer swap. |
| Verification rule | Verify that the entity is actually swapped and the replacement creates a role/belonging contradiction while remaining readable. |
| Rejection rule | Reject if the original and replaced entities are too similar, same-role peers, or if the rewrite leaves unresolved context conflicts. |
| Your logic | Just swapping one entity with another same-type entity does not make much sense. The replacement should differ in belonging or role. |

### Example 2

| Item | Text |
|---|---|
| Original selected sentence | এডিবি করোনা মোকাবেলায় বাংলাদেশের জন্য ৩ লাখ ডলারের জরুরি সহায়তা অনুমোদন করেছে। |
| Candidate rewrite | বিশ্ব ব্যাংক করোনা মোকাবেলায় বাংলাদেশের জন্য ৩ লাখ ডলারের জরুরি সহায়তা অনুমোদন করেছে। |
| Context risk | If headline or next sentence still says `এডিবি`, this rewrite creates contradiction. |
| Default implementation idea | Allow only if original entity appears only in target sentence; otherwise reject or update all linked mentions. |
| Why this works or fails | This works structurally, but `বিশ্বব্যাংক` and `এডিবি` are similar development-finance entities. Replacing with a wrong-role entity can create a stronger misinformation case. |
| Your final rule for this kind of case | Replace with something that should not naturally be here, while preserving coherent Bangla and updating linked mentions if needed. |

## Family 3: Temporal Shift

Temporal rewrites change a date, month, year, fiscal year, deadline, reporting period, or time anchor.

| Field | Your instruction |
|---|---|
| Allow | Allow contradictory changes to dates, months, years, fiscal years, reporting periods, deadlines, and time anchors. |
| Skip | Skip numeric amounts, percentages, counts, and rates; those belong to `numerical_fact`. Skip vague time expressions if they do not create a clear contradiction. |
| Rewrite method | Rewrite in a manner that changes the time frame while keeping the sentence coherent. Prefer exact span replacement when possible. |
| Replacement rule | Replace months, years, fiscal years, or dates in a way that contradicts the surrounding context or fixed event meaning. |
| Verification rule | Verify that the original time anchor is changed, the replacement appears, and the rewritten claim no longer has the same temporal meaning. |
| Rejection rule | Reject if the time meaning is unchanged, if the change is only formatting, or if the rewrite breaks Bangla grammar. |
| Your logic | Temporal misinformation should shift the time frame enough to create a contradiction, not merely restate the same time in another format. |

### Example 3

| Item | Text |
|---|---|
| Original selected sentence | জাতীয় শোক দিবস উপলক্ষে গত ১৫ আগস্ট বিজেএমসির সম্মেলন কক্ষে এক আলোচনা সভা ও দোয়া মাহফিল অনুষ্ঠিত হয়। |
| Candidate rewrite | জাতীয় শোক দিবস উপলক্ষে গত ২৫ আগস্ট বিজেএমসির সম্মেলন কক্ষে এক আলোচনা সভা ও দোয়া মাহফিল অনুষ্ঠিত হয়। |
| Context risk | `জাতীয় শোক দিবস` is semantically tied to `১৫ আগস্ট`, so changing only the date may be implausible or too obvious. |
| Default implementation idea | Exact span replacement only when the date is not culturally or legally fixed by surrounding words. |
| Why this works or fails | This is okay because changing `১৫ আগস্ট` to `২৫ আগস্ট` contradicts a fixed commemorative date while preserving the sentence structure. |
| Your final rule for this kind of case | Allow contradictory time shifts when they remain coherent and clearly change the factual time frame. |

## Family 4: Policy Reversal

Policy rewrites reverse approval, rejection, increase, decrease, implementation, withdrawal, or regulatory direction.

| Field | Your instruction |
|---|---|
| Allow | Allow reversal of approval, rejection, increase, decrease, implementation, suspension, withdrawal, barrier removal, or regulatory direction. |
| Skip | Skip vague policy discussion, claims without clear direction, or rewrites that cannot be expressed in coherent Bangla. |
| Rewrite method | Prefer controlled phrase-level reversal. Use LLM rewrite only when direct phrase replacement is not enough. |
| Replacement rule | Replace the policy direction with its opposite while preserving the same actor, topic, and source attribution where possible. |
| Verification rule | Verify that the rewritten policy direction contradicts the original direction and that the policy topic remains the same. |
| Rejection rule | Reject if the original policy direction remains, if the contradiction is vague, or if the Bangla is unnatural. |
| Your logic | Policy misinformation should reverse the main policy direction, not merely paraphrase or make the sentence less clear. |

### Example 4

| Item | Text |
|---|---|
| Original selected sentence | অবশেষে বহুল প্রত্যাশিত ব্যাংক ঋণের সুদের হার সিঙ্গেল ডিজিট ও সরল সুদ চালুর বাধা কাটল। |
| Candidate rewrite | অবশেষে বহুল প্রত্যাশিত ব্যাংক ঋণের সুদের হার সিঙ্গেল ডিজিট ও সরল সুদ চালুর বাধা আরও বেড়েছে। |
| Context risk | Policy reversal needs grammatical Bangla and clear directional opposition. Vague generated phrases should be rejected. |
| Default implementation idea | Use LLM only for short directional sentences; reject long attribution-heavy claims. |
| Why this works or fails | This works if `বাধা কাটল` becomes a clear opposite such as `বাধা আরও বেড়েছে`. It fails if the rewrite becomes vague or ungrammatical. |
| Your final rule for this kind of case | Reverse the main policy direction with a clear opposite phrase and preserve the rest of the sentence as much as possible. |

## Family 5: Causal Inversion

Causal rewrites change cause-effect claims such as `কারণে`, `ফলে`, `এর ফলে`, or `যার কারণে`.

| Field | Your instruction |
|---|---|
| Allow | Allow causal contradictions. |
| Skip | Skip if the original and rewritten meaning remain the same, or if the cause/effect pair is not clear. |
| Rewrite method | Rewrite with contradictory causal meaning. Do not merely swap sentence order. Preserve the cause when possible and change the effect into an economically opposite or implausible effect. |
| Replacement rule | Replace the effect in a style where it should not happen naturally in the given financial logic. |
| Verification rule | The rewritten sentence must contradict the original causal meaning. The causal marker should remain present and the cause should remain identifiable. |
| Rejection rule | Reject if the rewrite is same-meaning, if it only flips surface order, if cause/effect is unclear, or if Bangla is incoherent. |
| Your logic | Only contradiction makes causal inversion logical. The intended operation is flipping the main economic concept, not merely flipping sentence order. |

### Example 5

| Item | Text |
|---|---|
| Original selected sentence | আমদানি ব্যয় বেড়ে যাওয়ার কারণে বৈদেশিক মুদ্রার রিজার্ভে চাপ সৃষ্টি হয়েছে। |
| Candidate rewrite | আমদানি ব্যয় বেড়ে যাওয়ার কারণে বৈদেশিক মুদ্রার রিজার্ভ বেড়েছে। |
| Context risk | Causal inversion can create unsupported or nonsensical causality. This family has high hallucination risk. |
| Default implementation idea | Keep the cause mostly unchanged and change the effect to the opposite financial outcome. |
| Why this works or fails | This works because import cost increase normally creates pressure on foreign reserves, not reserve growth. The rewrite flips the main economic implication. |
| Your final rule for this kind of case | Do not flip only the sentence order. Flip the main concept: if import cost rises and the original says reserves are pressured, rewrite it as import cost rising caused reserves to increase. |

## Priority Order

Edit this order if you want the pipeline to prefer some rewrite families over others.

1. numerical_fact:
2. causal_inversion:
3. entity_replacement:
4. temporal_shift:
5. policy_reversal:

## Final Implementation Notes

Use this section to give explicit implementation choices.

| Decision | Your instruction |
|---|---|
| Families to enable for next 10-sample run | numerical_fact, causal_inversion, entity_replacement, temporal_shift, policy_reversal |
| Families to disable for now | None specified. |
| Should numerical rewrites use deterministic replacement? | Yes when the target span and planned replacement are known; replacement must create meaningful scale contradiction. |
| Should temporal rewrites use deterministic replacement? | Yes when the target time span and replacement are known. |
| Should entity rewrites be rejected if entity appears in headline/context? | Reject only if linked headline/context cannot be updated consistently. Otherwise update linked mentions. |
| Should policy rewrites require manual review? | Not always, but reject vague or incoherent policy reversals. |
| Should causal rewrites be disabled? | No. Enable, but require clear causal contradiction and coherent Bangla. |
| Any Bangla-specific spelling/style rules | Preserve unchanged spelling and source attribution. Rewritten Bangla must be coherent and close to the original sentence length. |
