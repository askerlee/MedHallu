# Priority-Ordered Rule Engine Spec (Pancreatic PET Guideline)

## 1) Goal
Implement deterministic, auditable recommendation logic for PET/PET-CT use in pancreatic cancer scenarios, with rule-priority conflict resolution and mandatory safety text.

## 2) Data Contract

### Required Inputs
- clinical_scenario: one of
  - INITIAL_DIAGNOSIS
  - PREOP_STAGING
  - TREATMENT_RESPONSE
  - RECURRENCE
  - RESTAGING
  - SOLITARY_METASTASIS_AT_RECURRENCE
- surgical_candidacy: YES | NO | UNKNOWN
- biopsy_status: DIAGNOSTIC | INCONCLUSIVE | NOT_FEASIBLE | NOT_DONE | UNKNOWN
- recurrence_status: YES | NO | UNKNOWN

### Optional Inputs
- positive_pet_would_change_to_curative_resection: YES | NO | UNKNOWN
- suspected_tumor_type: ADENOCARCINOMA | NEUROENDOCRINE | OTHER | UNKNOWN
- post_resection_status: COMPLETE_RESECTION | INCOMPLETE_RESECTION | NONE | UNKNOWN
- care_context: ROUTINE | CLINICAL_TRIAL | UNKNOWN

## 3) Output Contract
- recommendation_class:
  - RECOMMENDED
  - NOT_RECOMMENDED
  - CONDITIONAL_CONSIDER
  - NO_RECOMMENDATION
  - INSUFFICIENT_DATA
- activated_rules: ordered list of rule IDs
- rationale: concise clinician-facing explanation
- qualifiers: list of appended qualifiers
- safety_notice: mandatory final statement
- guideline_concordance: CONCORDANT | DISCORDANT | UNDETERMINED
- missing_fields: list (if INSUFFICIENT_DATA)

## 4) Rule Priority and Execution Model

### Deterministic Ordering
Evaluate rules in descending numeric priority. Higher priority wins for conflicting recommendation_class.

### Rule Types
- GATE: blocks downstream recommendation (example: missing required data)
- DECISION: sets recommendation_class
- MODIFIER: appends caveats, warnings, or constraints
- WRAPPER: always append global evidence and safety statements

### Conflict Resolution
1. If any GATE rule fires, final recommendation_class = INSUFFICIENT_DATA and stop DECISION evaluation.
2. Among DECISION rules, the highest-priority fired rule sets recommendation_class.
3. DECISION ties at same priority resolve by specificity score:
   - more conjunctive predicates wins (more specific)
4. MODIFIER rules never replace recommendation_class; they only append qualifiers.
5. WRAPPER rules always execute at end.

## 5) Priority Ladder (Highest to Lowest)
1. R01 Missing critical inputs (GATE, priority 100)
2. R02 Initial diagnosis default not recommended (DECISION, 90)
3. R03 Initial diagnosis exception for inconclusive/not feasible biopsy + direct surgical impact (DECISION, 85)
4. R04 Neuroendocrine FDG-negative caution (MODIFIER, 80)
5. R05 Pre-op staging in curative surgery candidates recommended (DECISION, 70)
6. R06 Pre-op staging non-candidate not recommended (DECISION, 65)
7. R07 Treatment response insufficient evidence (DECISION, 60)
8. R08 Incomplete resection response not recommended (DECISION, 58)
9. R09 Recurrence/restaging routine not recommended (DECISION, 50)
10. R10 Recurrence/restaging clinical-trial-only consideration (DECISION, 48)
11. R11 Solitary metastasis at recurrence no recommendation (DECISION, 45)
12. R12 Global evidence-quality qualifier (WRAPPER, 20)
13. R13 Global safety notice and override logging (WRAPPER, 10)

## 6) Execution Pseudocode

```text
function evaluate(input):
  fired = []
  qualifiers = []
  recommendation = null

  if R01(input):
    fired.append("R01")
    return output(
      recommendation_class="INSUFFICIENT_DATA",
      missing_fields=find_missing_required(input),
      activated_rules=fired + ["R12", "R13"],
      qualifiers=[evidence_note(), safety_note()],
      guideline_concordance="UNDETERMINED"
    )

  # Evaluate decision rules by descending priority
  decision_candidates = []
  for rule in [R02, R03, R05, R06, R07, R08, R09, R10, R11]:
    if rule.condition(input):
      fired.append(rule.id)
      decision_candidates.append(rule)

  if decision_candidates is empty:
    recommendation = "NO_RECOMMENDATION"
  else:
    recommendation = select_highest_priority_then_specificity(decision_candidates).class

  # Modifiers
  if R04(input):
    fired.append("R04")
    qualifiers.append("FDG-negative neuroendocrine disease possible; negative PET does not exclude disease")

  # Wrappers (always)
  fired.append("R12")
  qualifiers.append(evidence_note())
  fired.append("R13")
  qualifiers.append(safety_note())

  return output(
    recommendation_class=recommendation,
    activated_rules=stable_order(fired),
    qualifiers=qualifiers,
    guideline_concordance="CONCORDANT"
  )
```

## 7) Specificity Scoring (for tie-breaking)
Specificity score = count of non-trivial predicates in activation condition.
- Example:
  - R03 has 3 major predicates -> score 3
  - R02 has 1 predicate -> score 1
Therefore, if both fire in the same class, R03 wins.

## 8) Audit and Governance Requirements
- Log input snapshot, activated rules, final recommendation, and timestamp.
- Log clinician override decisions with reason.
- Version the policy package (semantic versioning recommended).
- Revalidate on any rule or priority update.

## 9) Example Engine Configuration

```yaml
policy_id: pancreatic-pet-guideline
policy_version: 1.0.0
default_recommendation: NO_RECOMMENDATION
evaluation_mode: priority_desc
tie_breaker: specificity_then_rule_id
always_append_rules:
  - R12
  - R13
gate_rules:
  - R01
```
