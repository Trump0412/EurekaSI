# TIP support eligibility

Multi-image input alone is not sufficient for TIP. A sample may have no
cross-frame tracks above the fixed confidence/visibility thresholds, or may lack
a valid same-frame non-neighbor substitution. These are objective eligibility
conditions, not reasons to fabricate edges or weaken confidence thresholds.

`scripts/prepare-tip-support.py` audits the actual cached graph on each rank.
It retains an ID-level record with eligibility, edge count and exclusion reason;
merge requires full manifest coverage and complete shard receipts. Single-frame
examples are explicitly ineligible. Unexpected graph/IO/model errors still fail.

The stage worker consumes the accepted IDs for TIP bootstrap and TIP replay.
Instruction SFT keeps every original example, original ordering rule, global64,
and the existing explicit tail-padding policy. This narrows TIP's population from
all multi-frame examples to supported examples and must be reported in training
budgets. It does not change SFT's source manifest. Comparisons between GeoRoute
arms must use the same eligibility protocol, with post-merger support audited in
the actual coarsened graph space. Actual frame count/order remain unchanged.

Support preparation is real frozen-teacher computation, not model training.
It can be expensive on a cold cache. Its throughput must not be used as the
training ETA. The allocation owner can request a yield between examples for
priority RFT, then resume from flushed ID records.

Separately, the mask sampler now rejects a new destination if hiding it would
remove the last visible incoming source of a previously accepted destination.
The final support assertions remain enabled. A passing CPU test is not full GPU
acceptance: graph audit, real updates, checkpoint reload and influence checks
must all pass before formal training.
