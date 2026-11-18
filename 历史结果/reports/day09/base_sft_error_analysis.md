# Day09 - Base vs SFT Error Analysis

## 1. Evaluation setup

Both models are evaluated on the same fixed dev-500 split with the same:

- prompt protocol
- decoding settings
- answer parser
- strict evaluator

Compared models:

- Base: Qwen3-1.7B-Base
- SFT: Qwen3-1.7B-Base + final LoRA adapter

Formal dev-500 results:

| Model | Correct | Wrong | Format fail | Strict accuracy | Format success |
|---|---:|---:|---:|---:|---:|
| Base | 267 | 64 | 169 | 53.40% | 66.20% |
| SFT | 390 | 108 | 2 | 78.00% | 99.60% |

The strict accuracy increased by 24.60 percentage points, but this increase should not be interpreted entirely as pure mathematical reasoning improvement.

## 2. Paired status transitions

The Base and SFT predictions were aligned by sample ID.

| Base status | SFT status | Count |
|---|---|---:|
| correct | correct | 231 |
| correct | format_fail | 1 |
| correct | wrong | 35 |
| format_fail | correct | 131 |
| format_fail | format_fail | 1 |
| format_fail | wrong | 37 |
| wrong | correct | 28 |
| wrong | wrong | 36 |

Overall:

- Non-correct -> correct: 159 samples
- Correct -> non-correct: 36 samples
- Net increase in correct answers: 123 samples

This matches:

267 + 123 = 390 correct answers.

## 3. Main observations

### A. Format-following improvement

The largest transition is:

format_fail -> correct: 131 samples

In an initial manual sample of three cases, the Base model already produced the correct numerical answer, but failed the strict final-answer format. The SFT model retained the correct answer while following the required output format.

This suggests that a substantial part of the strict-accuracy improvement comes from better protocol and format compliance.

### B. Reasoning / problem-solving improvement

The transition:

wrong -> correct: 28 samples

contains examples where the Base model produced a parseable but mathematically incorrect answer, while the SFT model corrected the reasoning.

#### Case: ID 5600

Reference answer: 845640

Base:
- Produced 593319.
- Failed to correctly complete the population calculation.

SFT:
- Computed Greenville as 482653 - 119666 = 362987.
- Computed total population as 482653 + 362987 = 845640.

Observed improvement:
- multi-step calculation / reasoning correction

#### Case: ID 3850

Reference answer: 30

Base:
- Treated the four pyramid levels as 1 + 2 + 3 + 4.
- Produced 10.

SFT:
- Recognized the levels as square layers:
  1^2 + 2^2 + 3^2 + 4^2
- Produced 30.

Observed improvement:
- problem interpretation / structural reasoning

#### Case: ID 4307

Reference answer: 360000

Base:
- Correctly computed monthly rent as 30000.
- Failed to multiply by 12 months.
- Produced 30000.

SFT:
- Added the missing yearly conversion:
  30000 * 12 = 360000.

Observed improvement:
- missing-step recovery / multi-step completeness

### C. SFT regression

The transition:

correct -> wrong: 35 samples

shows that SFT did not improve every example.

#### Case: ID 1184

Reference answer: 320

Base:
- Correctly accounted for two large beds and two medium beds.
- Produced 320.

SFT:
- Calculated only one large bed and one medium bed.
- Produced 160.

Observed regression:
- quantity-condition omission
- loss of multiplicity information

This shows that SFT can improve format compliance and many reasoning cases while still causing regressions on some examples.

## 4. Current conclusion

The SFT gain from 53.40% to 78.00% is a mixture of:

1. strong improvement in strict output-format compliance
2. genuine reasoning / calculation improvements on some samples
3. regressions on a smaller but non-trivial set of previously correct samples

Therefore, the +24.60 percentage-point strict-accuracy improvement should not be described as a pure +24.60 percentage-point reasoning improvement.

Further manual error analysis is required before making stronger claims about reasoning capability.

## 5. Manual audit of 20 representative cases

A fixed 20-case manual audit set was constructed across several Base-to-SFT status transitions.

Manual labels:

| Manual label | Count |
|---|---:|
| format_only | 6 |
| reasoning_improved | 4 |
| still_wrong | 4 |
| condition_omission | 3 |
| arithmetic_error | 2 |
| problem_misinterpretation | 1 |
| **Total** | **20** |

### Interpretation

**format_only**

In these cases, the Base model already reached the correct numerical result but failed the strict final-answer protocol. SFT mainly improved output-format compliance.

**reasoning_improved**

These cases show genuine improvement in problem solving, such as fixing multi-step quantity tracking, relation interpretation, or missing reasoning steps.

**still_wrong**

These cases show that fixing output format does not guarantee mathematical correctness. Some SFT outputs became parseable while remaining incorrect, and in some cases the Base model had an implicitly correct answer before SFT introduced a reasoning error.

**condition_omission**

Several regressions came from dropping or misreading problem constraints, such as multiplicity, fractions, date boundaries, or changing state across steps.

**arithmetic_error**

Some errors were caused by incorrect arithmetic despite an otherwise plausible reasoning structure.

**problem_misinterpretation**

At least one inspected regression came from misunderstanding the underlying structure of the problem rather than from simple arithmetic.

### Important limitation

This 20-case set is a stratified diagnostic sample selected across transition categories. It is not a random sample from the full dev-500 set.

Therefore, label proportions in this manual audit must not be interpreted as population-level error rates.

The audit is used to identify representative failure modes and explain the mechanisms behind the aggregate Base-to-SFT metric changes.
