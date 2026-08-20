# Fix minimi prima di mandare il PDF ai coautori — 2026-08-19

Solo quello che esce rotto nel PDF. Niente riscritture di prosa.

---

## A. Escono come `??` o `[?]` nel PDF

### A1. Riferimento a sezione inesistente
In §Experimental setup, "Configuration", ultimo capoverso.

TROVA     Section~\ref{sec:sensitivity} varies the sample size and the
SOSTITUISCI Section~\ref{sec:ablation} varies the sample size and the

(`\label{sec:sensitivity}` vive solo in `snippet_sensitivity.tex`, sezione mai
inserita nel paper.)

### A2. Filename figura con spazi e parentesi
`\includegraphics` con spazi/parentesi nel nome fallisce o si comporta in modo
imprevedibile. Rinomina il file su Overleaf in `fig1_generator.png`, poi:

TROVA     \includegraphics[width=0.73\textwidth]{shift-surv (10).png}
SOSTITUISCI \includegraphics[width=0.73\textwidth]{fig1_generator.png}

### A3. Sei cite key da controllare
Nel .bib LOCALE mancano. Tre esistono con nome diverso — se su Overleaf le hai
gia' rinominate, ignora; altrimenti allinea paper e bib:

  paper usa                 bib locale ha
  shaker2023multisource  →  shaker2022multisource
  ye2025multicalibration →  ye2024multicalibration
  guo2022evaluation      →  guo2021evaluation

Tre non esistono in nessuna variante. Entry pronte da incollare in
references.bib:

@book{westfall1993resampling,
  author    = {Westfall, Peter H. and Young, S. Stanley},
  title     = {Resampling-Based Multiple Testing: Examples and Methods for
               $p$-Value Adjustment},
  publisher = {Wiley},
  address   = {New York},
  year      = {1993}
}

@article{romano2005exact,
  author  = {Romano, Joseph P. and Wolf, Michael},
  title   = {Exact and approximate stepdown methods for multiple hypothesis
             testing},
  journal = {Journal of the American Statistical Association},
  volume  = {100},
  number  = {469},
  pages   = {94--108},
  year    = {2005},
  doi     = {10.1198/016214504000000539}
}

@article{gretton2012kernel,
  author  = {Gretton, Arthur and Borgwardt, Karsten M. and Rasch, Malte J. and
             Sch{\"o}lkopf, Bernhard and Smola, Alexander},
  title   = {A kernel two-sample test},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  pages   = {723--773},
  year    = {2012}
}

---

## B. Visibili come errori nel testo

### B1. Punti interrogativi letterali nel modello generativo

TROVA     where $C_d \in \mathbb{R}^{? \times ?}$ controls
SOSTITUISCI where $C_d \in \mathbb{R}^{p^{\mathrm{inv}} \times p^{\mathrm{spu}}}$ controls

### B2. Simbolo sbagliato

TROVA     $u_{i,d}^{\mathrm{spu}} \in \mathbb{R}^{d^{spu}}$
SOSTITUISCI $u_{i,d}^{\mathrm{spu}} \in \mathbb{R}^{p^{\mathrm{spu}}}$

### B3. Manca la variabile

TROVA     only on the invariant factors $_{i,d}^{\mathrm{inv}}$
SOSTITUISCI only on the invariant factors $u_{i,d}^{\mathrm{inv}}$

### B4. Frase troncata

TROVA     where $\beta_{0,d}$ and $w_d$.
SOSTITUISCI where $\beta_{0,d}\in\mathbb{R}$ is a center-specific intercept and
            $w_d\in\mathbb{R}^{p^{\mathrm{inv}}}$ the latent risk coefficients.

### B5. Conteggio sbagliato ("both" ma i regimi sono tre)
§Results, calibrazione, penultimo capoverso.

TROVA     baseline-hazard, concept shift and label-prior shift with $p=0.001$ in both regimes.
SOSTITUISCI baseline-hazard, label-prior, and concept shift, with $p=0.001$ in each of the three.

### B6. Il check Cox e' promesso ma non c'e'
Abstract e Introduction annunciano un'analisi Cox sul follow-up censurato che
in §V (real data) non esiste. Per la versione ai coautori la mossa piu' rapida
e' togliere le due promesse; il paragrafo vero e' gia' scritto in
REVISION_2026-08-19.tex e si reinserisce quando i numeri sono riverificati.

ABSTRACT — togli l'ultima frase:

TROVA      in AUROC over ERM. Cox analysis using the full censored follow-up
           reproduces the source-to-target degradation, indicating that it is
           not an artifact of the fixed-horizon dichotomization.
SOSTITUISCI in AUROC over ERM.

INTRODUCTION — accorcia la frase sul binary endpoint:

TROVA      the event times are retained and used to verify, in the real
           cohorts, that the transport conclusions do not depend on this
           dichotomization (Section~\ref{sec:real}).
SOSTITUISCI the underlying event times are retained by the generator.

CONTRIBUZIONI — terza voce:

TROVA      on two real multi-center clinical cohorts, with the real-cohort
           conclusion verified using the full censored follow-up.
SOSTITUISCI on two real multi-center clinical cohorts.

---

## C. Typo (30 secondi, tutti find→replace singoli)

  This section describe configuration      → This section describes the configuration
  Table~\ref{tab:design} summerize          → Table~\ref{tab:design} summarizes
  we comment the results on real worl       → we comment the results on real-world
  Neither Adaptation and Generalization recover
                                            → Neither adaptation nor generalization recovers
  Neither procedures remove                 → Neither procedure removes
  $I_p$ is the indicator matrix             → $I_p$ is the identity matrix
  a domain-specific observation maps        → domain-specific observation maps

---

## Non toccato di proposito

- Prosa di abstract/intro/discussion: la riscrittura sta in
  REVISION_2026-08-19.tex, da applicare dopo il giro con i coautori.
- Conteggio pagine (stimato 10-11 contro un limite di 10 incluse le reference).
- Sezione sensitivity, statement di riproducibilita'.
