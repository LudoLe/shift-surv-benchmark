# Allineamento poster ESAC-26 ↔ SHIFT-SURV (stato attuale)

Confronto tra `shift-surv-esac-26 (7).pdf`, il `README.md` e la presentazione
`SHIFT-SURV-ADAPTATION.pptx` (26/06/26) + codice (`generator/parameters.py`,
`experiments/run_adaptation.py`).

Il poster riflette una **versione precedente** del lavoro. Lo stato attuale ha due
elementi cardine che nel poster **mancano del tutto**:
1. lo split **discrimination vs calibration** ("un AUROC stabile può nascondere drift di calibrazione");
2. la **validazione su dati reali I3LUNG** (7 transfer).

---

## A. Errori fattuali da correggere (sicuri)

| # | Dove | Poster (sbagliato) | Corretto | Fonte |
|---|------|--------------------|----------|-------|
| A1 | Fig. 5 caption | "Target AUROC at **12 months**" | "Target AUROC **at τ = 6 months**" | `parameters.py:67` `tau = 6.0` |
| A2 | Introduzione / Methods | "**Cox–Weibull** generator" | "**Weibull proportional-hazards** generator" | Cox è semiparametrico; il DGP è Weibull PH parametrico (README, slide 4) |
| A3 | Fig. 3 titolo | "Concept **Drift**" | "Concept **shift**" (coerenza coi nomi scenari) | `scenarios.py` / README usano `concept_shift` |

---

## B. Lacune di contenuto (il poster non racconta il lavoro attuale)

### B1. Lista metodi incompleta → riscrivere
Il poster cita solo **ERM, CORAL, IRM, DANN, CDAN**. Il benchmark attuale gira **10 metodi**
(`run_adaptation.py:97-106`), e i metodi di **calibrazione** sono centrali per la storia attuale:

- Invariant oracle (upper bound)
- LR-ERM (pooled) + MLP-ERM
- IW-ERM (importance weighting)
- CORAL, IRM
- DANN, CDAN
- **Prior recalibration (Saerens EM)**, **Platt scaling**, **Isotonic** ← calibrazione post-hoc

> Senza i metodi di calibrazione, la conclusione attuale ("la calibrazione è
> parzialmente e selettivamente recuperabile") non ha supporto sul poster.

### B2. Risultati: solo 3 shift descritti su 7 core
I bullet RESULTS coprono solo **Covariate / Concept / Hazard**, ma il benchmark core ha
**7 scenari**: `no_shift, covariate_shift, observational_shift (Measurement), spurious_shift,
baseline_hazard_shift, concept_shift, shape_shift`. Le figure (Fig. 4/5) già ne mostrano di più
del testo → disallineamento testo/figure. Aggiungere almeno Measurement e Spurious ai bullet.

### B3. Manca tutta la validazione su dati reali (I3LUNG)
Il messaggio chiave attuale (slide 1, 11, 12) è: *"lo stesso pattern
discrimination/calibration si riproduce su 7 transfer reali I3LUNG"*. Il poster è **solo
sintetico**. Serve un blocco/figura dedicata ai 7 transfer I3LUNG, altrimenti manca il
contributo principale. (Asset già pronto: `paper_assets/fig4_realdata.pdf`.)

### B4. Conclusione obsoleta → riscrivere
La conclusione attuale è imperniata sullo split AUROC/ECE e sull'evidenza reale, non sul
comportamento per-singolo-shift. Vedi testo pronto in §C.

---

## C. Testo pronto da incollare (allineato allo stato attuale)

**TITOLO** (opzione allineata alla domanda di ricerca della talk):
> SHIFT-SURV: A Synthetic Multi-Centre Generator for Evaluating Unsupervised Domain
> Adaptation in Fixed-Horizon Cancer Survival Risk

**INTRODUCTION**
- I modelli di rischio/sopravvivenza addestrati in un centro **degradano** quando trasferiti
  altrove (case mix, protocolli di misura, percorsi di cura, rischio di base).¹˒²
- SHIFT-SURV è un generatore sintetico **Weibull proportional-hazards** multi-centro che
  separa componenti **invarianti, spurie e di rumore**.
- Domanda: *l'unsupervised domain adaptation recupera davvero rischio trasportabile sotto
  shift multi-centro — e come lo verifichiamo?*
- Banco di prova per ERM, IW-ERM, CORAL, IRM, DANN, CDAN **e calibrazione post-hoc**
  (prior recalibration, Platt).³⁻⁶

**METHODS** (aggiungere ai metodi DA esistenti)
- Calibrazione post-hoc: **Prior recalibration (Saerens EM)**, **Platt scaling**.
- Protocollo: 1 source + 1 target, τ = **6 mesi**, label y = 1[T ≤ τ], 10 seed, media ± 95% CI.
- Metriche: **AUROC** (discriminazione) **ed ECE** (calibrazione), gap source→target.

**RESULTS** (sostituire i 3 bullet con la cornice attuale)
- **Discriminazione e calibrazione divergono**: ogni shift induce combinazioni diverse di
  perdita di AUROC e di errore di calibrazione recuperabile — degradano in modo indipendente.
- **Nessun recupero pratico dell'AUROC**: best-DA ≈ ERM ovunque (t-test appaiato ns), e
  **peggiore su Spurious**; ampio headroom verso l'oracolo resta non catturato.
- **Calibrazione parzialmente/selettivamente recuperabile**: migliora su shift feature-driven
  (Covariate **, Measurement *) ma non su baseline-hazard/concept.
- **Conferma su dati reali (I3LUNG, 7 transfer)**: l'AUROC non migliora mai su ERM
  (significativamente peggiore su 4/7); la calibrazione migliora su diversi transfer.

**CONCLUSION** (sostituire integralmente)
- Nel regime testato (small-sample, multi-centro) la domain adaptation non supervisionata
  **non recupera l'AUROC** rispetto a ERM, nonostante ampio headroom dell'oracolo.
- Discriminazione e calibrazione **degradano in modo indipendente**: un AUROC stabile può
  **mascherare un drift di calibrazione sostanziale**.
- Lo stesso pattern qualitativo **si riproduce su 7 transfer reali I3LUNG**: l'adattamento non
  migliora la discriminazione, ma può aiutare **selettivamente** la calibrazione.
- SHIFT-SURV è utile come **benchmark controllato per attribuire il fallimento di trasporto a
  meccanismi di shift specifici**, non come prova che un singolo metodo DA "vinca".

---

## D. Da decidere (tensione tra poster/figure e talk)

- **IRM**: il codice lo esegue e le figure del poster lo mostrano, ma la slide 13 (limiti) lo
  dichiara *out of scope* (richiede ≥2 environment, qui c'è 1 source–1 target). Decidere se
  tenerlo come metodo "anomalo (single-source)" o spostarlo in appendice/limiti. Evitare che
  poster e talk si contraddicano.
- **Claim "CDAN amplifica lo shortcut sotto spurious shift"**: la talk dice più genericamente
  "best-DA peggiore su Spurious". Verificare nei risultati se la specifica attribuzione a CDAN
  regge, altrimenti generalizzare.
- **Nome scenario**: "observational_shift" (codice/README) = "Measurement" (talk). Scegliere
  un solo termine nel poster.
