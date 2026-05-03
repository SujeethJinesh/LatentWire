# HellaSwag Hybrid Anti-Harm Veto Gate

- positive method pass: `False`
- total rows: `9216`
- fit / selection rows: `512` / `512`
- heldout eval rows: `8192`
- candidate-only accuracy: `0.525499`
- fixed hybrid accuracy: `0.531141`
- main veto accuracy: `0.529663`
- main veto delta vs fixed hybrid: `-0.003052`
- main veto CI95 low vs fixed hybrid: `-0.005249`
- main veto avoided harms / missed hybrid helps: `32` / `57`
- candidate/hybrid oracle accuracy: `0.539062`
- cross-family pass: `False`
- cross-family veto accuracy: `0.462240`
- cross-family veto delta vs fixed hybrid: `-0.005208`

## Interpretation

The anti-harm veto gate tests a selective-classification style source-side accept/fallback rule for the current fixed hybrid packet. A pass would strengthen the harm-controlled packet method without changing the 1B raw / 4B framed receiver-visible contract. A failure means the current shallow packet features do not reliably separate hybrid helps from hybrid harms, so the next method branch should move to a real receiver/common-basis signal rather than another shallow veto.

## Lay Explanation

The hybrid hint sometimes changes the answer choice and occasionally makes it worse. This test uses the first half-slice to define simple warning rules, the next half-slice to pick one, then freezes it. When a later switch looks risky, the rule keeps the old hint; otherwise it uses the hybrid hint.
