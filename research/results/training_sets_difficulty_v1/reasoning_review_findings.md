# Assistant reasoning review — October 12, 2026

I read each of the 20 selected model solutions and reconstructed its arithmetic from the question. This is an assistant review, not an independent human annotation. Original examples, selected candidates and group files are unchanged; no training occurred.

**Findings:** 15 explanations appear sound; two contain clear cancelling errors (indices 1221 and 613); one has substantive but repairable/borderline bookkeeping defects (807); two have minor wording/unit-label defects requiring a human decision (1522 and 879). All three more serious cases are among the 15 sampled 1/4 problems. This deliberately stratified sample does not estimate the whole pool’s error rate.

| Review | Index | Bucket | Assessment |
|---:|---:|---:|---|
| 1 | 1530 | 1/4 | sound |
| 2 | 1074 | 1/4 | sound |
| 3 | 1221 | 1/4 | substantive_error |
| 4 | 578 | 1/4 | sound |
| 5 | 1071 | 1/4 | sound |
| 6 | 1522 | 1/4 | minor_wording_human_review |
| 7 | 1399 | 1/4 | sound |
| 8 | 613 | 1/4 | substantive_error |
| 9 | 924 | 1/4 | sound |
| 10 | 901 | 1/4 | sound |
| 11 | 974 | 1/4 | sound |
| 12 | 900 | 1/4 | sound |
| 13 | 717 | 1/4 | sound |
| 14 | 1499 | 1/4 | sound |
| 15 | 807 | 1/4 | substantive_error_borderline_bookkeeping |
| 16 | 899 | 3/4 | sound |
| 17 | 982 | 2/4 | sound |
| 18 | 833 | 3/4 | sound |
| 19 | 922 | 2/4 | sound |
| 20 | 879 | 3/4 | minor_wording_human_review |

## Detailed checks

### 1. Index 1530 — sound

138 animals initially; half remain (69), then 37 goats gives 106; half male gives 53.

### 2. Index 1074 — sound

20×60/3=400 songs monthly; 400×12×0.50=2400 yearly. Units and all steps agree.

### 3. Index 1221 — substantive_error

The outside is stated to take twice the house time, but the solution assigns it only the house time. Two siblings plus James means three workers, but the solution divides by two. These errors cancel under the reference’s intended household-time interpretation: house 240, outside 480, total 720, divided by three = 240 minutes. The wording about two bathrooms versus singular “the bathroom” is also ambiguous; reading 120 minutes per bathroom instead yields 6 hours. The model’s two explicit errors remain invalid under either reading.

### 4. Index 578 — sound

Monday 2, Tuesday 4, Friday 1, Thursday 4; intended remaining amount Wednesday is 20−11=9.

### 5. Index 1071 — sound

Desired membership 2×10+5=25, so additional members =25−10=15.

### 6. Index 1522 — minor_wording_human_review

The computation 9×(4+20+10)=306 is sound, but “each plot produces ...34” is false: plot totals are 36,180,90. The 34 is the combined yield of one plant of each type, not one plot. Treat as a unit-label/explanation defect, not a demonstrated arithmetic cancellation.

### 7. Index 1399 — sound

Seven original boxes contain 70 cards; add five unboxed cards =75. The intermediate 70 is the boxed subtotal.

### 8. Index 613 — substantive_error

70% cheaper means each visit costs 30% of 200 =60, not 140. Two visits cost 120; savings are 200−120=80. The model instead spends 280 and reverses the subtraction (280−200), incorrectly calling that savings. Two clear errors cancel numerically.

### 9. Index 924 — sound

Two payments of 757 give total expenses 1514; rent/utilities subtotal1214; groceries300. The first “total cost” sentence should be read as the fixed-cost subtotal, which the later subtraction correctly uses.

### 10. Index 901 — sound

After losing half there are15, then27; Thursday buys one-third of Tuesday’s15 =5, giving32.

### 11. Index 974 — sound

Each bank contains100×.01+50×.10=6 dollars; two banks give12.

### 12. Index 900 — sound

Three hats6 hours, three scarves9, six mittens6, six socks9, three sweaters18; total48.

### 13. Index 717 — sound

Two half-hour sessions plus one-fifth hour =1.2 hours =72 minutes.

### 14. Index 1499 — sound

Under the standard365-day-year convention used by the reference:730 desired annual inches, minus430 =300 over100 days, so3 per day.

### 15. Index 807 — substantive_error_borderline_bookkeeping

As written, “total ...16 ounces during the day” omits one of three8-ounce drinks, and “72 ... at the end” contradicts the given56. The new-container subtotal also omits dinner. Correct conservation is initial+64−24=56, so initial16. The model’s final formula is algebraically repairable by reinterpreting16 as breakfast-plus-dinner and72 as a hypothetical restored quantity, but its stated intermediate quantities are false. Human review should judge severity; this is a flawed explanation, not proof of accidental arithmetic cancellation.

### 16. Index 899 — sound

Together the two players score20%×300=60; equal shares30.

### 17. Index 982 — sound

Feed cost48000, combined cost88000, revenue100×1000×2=200000; profit112000.

### 18. Index 833 — sound

Lizard6, frog18, toad27; with gecko12 total63.

### 19. Index 922 — sound

Betty50, pack cost100, twenty packs2000. Equation, interpretation and arithmetic agree.

### 20. Index 879 — minor_wording_human_review

The arithmetic and class counts are correct:12 boys,13 girls,25 total,52% girls. “11 boys +1 girl =12 boys” is a wrong-word slip; the question adds a boy and the rest of the solution consistently uses that. Human decision needed on excluding minor explanation defects.

## Token-matching outcome

All three sets contain 300 unique problems. Both comparison groups tried 1,000 draws and failed the ±3% supervised-token rule. The saved random and easy groups are UNMATCHED DIAGNOSTIC files, not approved training inputs.

Hard: 32,879 supervised tokens. Closest random draw 330: 29,482 (−10.33%). Closest easy draw 778: 25,430 (−22.66%). Even the 300 longest fixed easy solutions total 30,995 (−5.73%), below the permitted lower bound 31,892.63. Easy matching is therefore impossible under the current fixed-candidate/300-problem protocol. Random matching is feasible in principle by extreme bounds, but was not found in the agreed 1,000 uniform draws.

Do not train until a consistent solution-quality policy and a revised or explicitly accepted token-budget comparison have been settled. Filtering only the reviewed hard examples would change the group asymmetrically; no examples were removed here.
