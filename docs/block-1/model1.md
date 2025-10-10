
# Model 1 — CLV (BG/NBD + Gamma–Gamma): Computation Cards (Whiteboard‑first)

**Time unit:** days • **Calibration end:** t_cal • **Horizon:** H

Each card is a tiny step in strict order. Only three cards require a compute helper (optimizer/series loop); everything else is pure algebra.
Legend: **Why/What/Where/When/How** → purpose ledger; **Inputs** → baton vars needed; **Algebra** → explicit math; **Output** → baton produced.

---

## Band S — Setup & Notation
**S1 [Atomic] — Calendar & horizon**
- Why fix anchors • What t_cal, H • Where project constants • When start • How choose & freeze
- Inputs: —
- Algebra: constants t_cal, H (days)
- Output: t_cal, H

**S2 [Atomic] — Log identities**
- Why stable math • What log‑Gamma/Beta • Where counts model • When fitting • How expand
- Inputs: u,v>0
- Algebra: ln B(u,v)=ln Γ(u)+ln Γ(v)−ln Γ(u+v)
- Output: identity

---

## Band A — RFM Foundations (from orders)
**A1 [Atomic] — First/last order times**
- Why anchor timeline • What t_{i1}, t_{i n_i} • Where orders • When start • How min/max
- Inputs: {t_{ik}}
- Algebra: t_{i1}=min_k t_{ik}; t_{i n_i}=max_k t_{ik}
- Output: t_{i1}, t_{i n_i}

**A2 [Atomic] — Counts**
- Why repeats for BG/NBD • What n_i, x_i • Where orders • When after A1 • How direct
- Inputs: {t_{ik}}
- Algebra: n_i = #orders; x_i = max(n_i−1, 0)
- Output: n_i, x_i

**A3 [Molecule] — Age & last‑purchase time**
- Why BG time triplet • What T_i, t_{x,i} • Where S1,A1 • When after A2 • How subtract
- Inputs: t_{i1}, t_{i n_i}, t_cal
- Algebra: T_i=t_cal−t_{i1} (≥0); t_{x,i} = (n_i≥1 ? t_{i n_i}−t_{i1} : 0)
- Check: 0 ≤ t_{x,i} ≤ T_i
- Output: T_i, t_{x,i}

**A4 [Molecule] — Mean margin per txn**
- Why GG uses z̄_i • What z̄_i • Where orders • When after A2 • How average
- Inputs: {m_{ik}} (cents)
- Algebra: z̄_i = (n_i>0 ? (Σ m_{ik})/n_i : 0)
- Output: z̄_i

**A5 [Component] — RFM pack**
- Why hand to models • What (x_i,T_i,t_{x,i},z̄_i) • Where A2–A4 • When end Band A • How collect
- Inputs: A2–A4
- Algebra: package record per i
- Output: (x_i,T_i,t_{x,i},z̄_i)

---

## Band B — BG/NBD Fit (estimate r, α, a, b)
**B1 [Atomic] — Log pieces per customer**
- Why stable likelihood • What ln A1..A4 • Where S2,A5 • When pre‑sum • How expand
- Inputs: x_i,T_i,t_{x,i}; trial r,α,a,b>0
- Algebra:
  ln A1^{(i)} = ln B(a,b+x_i) − ln B(a,b)
  ln A2^{(i)} = ln Γ(r+x_i) − ln Γ(r) + r ln α
  ln A3^{(i)} = −(r+x_i) ln(α+T_i)
  ln A4^{(i)} = 1_{x_i>0}[ ln B(a+1,b+x_i−1) − ln B(a,b) − (r+x_i) ln(α+t_{x,i}) ]
- Output: ln A1..A4

**B2 [Molecule] — Per‑customer log‑likelihood**
- Why build objective • What ln L_i • Where B1 • When pre‑sum • How log‑sum‑exp
- Inputs: ln A1..A4
- Algebra: ln L_i = ln( exp(lnA1+lnA2+lnA3) + exp(lnA2+lnA4) )
- Output: ln L_i

**B3 [Module] — Total log‑likelihood**
- Why optimizer target • What J(r,α,a,b) • Where B2 • When fit • How sum
- Inputs: {ln L_i}
- Algebra: J = Σ_i ln L_i
- Output: J

**B4 [Component] — Maximize J → (r̂,α̂,â, b̂) (Compute helper)**
- Why estimate BG/NBD params • What argmax • Where B3 • When after B3 • How constrained optimizer
- Inputs: J
- Algebra: (r̂,α̂,â,b̂) = argmax_{r,α,a,b>0} J
- Compute: run from multiple positive seeds; keep best solution
- Output: (r̂,α̂,â,b̂)

---

## Band C — BG/NBD Horizon Expectation E[N_H]
**C1 [Atomic] — Shorthand**
- Why prep horizon math • What A_i,B_i,C_i,z_i,pow_i • Where A5,B4,S1 • When after fit • How define
- Inputs: x_i,T_i,t_{x,i}; (r̂,α̂,â,b̂); H
- Algebra: A_i=r̂+x_i; B_i=b̂+x_i; C_i=â+b̂+x_i−1; z_i=H/(α̂+T_i+H); pow_i=((α̂+T_i)/(α̂+T_i+H))^{A_i}
- Output: A_i,B_i,C_i,z_i,pow_i

**C2 [Molecule] — Hypergeometric series 2F1(A,B;C;z) (Compute helper)**
- Why finite‑horizon adjust • What S_i≈2F1 • Where C1 • When per i • How recurrence
- Inputs: A_i,B_i,C_i,z_i
- Algebra loop: u0=1, S=u0; for j=1..: u_j = u_{j−1} * ((A_i+j−1)(B_i+j−1)/((C_i+j−1)j)) * z_i; S+=u_j; stop |u_j|<ε or j cap
- Output: S_i≈2F1(A_i,B_i;C_i;z_i)

**C3 [Component] — Expected future transactions**
- Why get counts forecast • What E_i[N_H] • Where C1–C2 • When after S_i • How closed form
- Inputs: A_i,B_i,C_i,z_i,pow_i,S_i,t_{x,i},T_i; (â,b̂,α̂)
- Algebra:
  E_i[N_H] = ((â+b̂+x_i−1)/(â−1)) * (1 − pow_i*S_i) + 1_{x_i>0} * (â/(b̂+x_i−1)) * ((α̂+T_i)/(α̂+t_{x,i}))^{A_i}
- Output: E_i[N_H]

---

## Band D — Gamma–Gamma Fit (estimate p, q, γ)
**D1 [Atomic] — Per‑customer monetary log‑likelihood**
- Why fit spend model • What ln L_i^{GG} • Where A5 • When x_i>0 • How expand
- Inputs: x_i>0, z̄_i>0; trial p>0, q>1, γ>0
- Algebra:
  ln L_i^{GG} = ln Γ(p x_i + q) − ln Γ(p x_i) − ln Γ(q) + (p x_i − 1) ln z̄_i + (p x_i) ln x_i + q ln γ − (p x_i + q) ln(γ + x_i z̄_i)
- Output: ln L_i^{GG}

**D2 [Molecule] — Total GG log‑likelihood**
- Why optimizer target • What K(p,q,γ) • Where D1 • When fit • How sum
- Inputs: {ln L_i^{GG}} over x_i>0
- Algebra: K = Σ_{i:x_i>0} ln L_i^{GG}
- Output: K

**D3 [Component] — Maximize K → (p̂, q̂, γ̂) (Compute helper)**
- Why estimate GG params • What argmax • Where D2 • When after D2 • How constrained optimizer
- Inputs: K
- Algebra: (p̂, q̂, γ̂) = argmax_{p>0,q>1,γ>0} K
- Compute: run from multiple positive seeds; keep best solution
- Output: (p̂, q̂, γ̂)

---

## Band E — Gamma–Gamma Expectation E[M]
**E1 [Molecule] — Expected margin per txn**
- Why per‑txn value • What E_i[M] • Where A5,D3 • When after GG fit • How rational form
- Inputs: (p̂,q̂,γ̂), x_i, z̄_i
- Algebra: E_i[M] = p̂ (γ̂ + x_i z̄_i) / (p̂ x_i + q̂ − 1)
- Output: E_i[M] (cents/txn)

**E2 [Component] — Monetary pack**
- Why hand to assembly • What E_i[M] • Where E1 • When end Band E • How collect
- Inputs: E_i[M]
- Algebra: package per i
- Output: E_i[M]

---

## Band F — Assembly & Tiers (Grand Component)
**F1 [Atomic] — Horizon CLV (cents)**
- Why final value • What CLV_{i,H} • Where C3,E2 • When endgame • How multiply
- Inputs: E_i[N_H], E_i[M]
- Algebra: CLV_{i,H} = E_i[N_H] * E_i[M]
- Output: CLV_{i,H} (cents)

**F2 [Molecule] — Unit conversion**
- Why presentability • What dollars • Where F1 • When optional • How divide
- Inputs: CLV_{i,H}
- Algebra: CLV^{$}_{i,H} = CLV_{i,H} / 100
- Output: CLV^{$}_{i,H}

**F3 [Module] — Quantile cutoffs**
- Why tier thresholds • What q_{0.9}, q_{0.5} (example) • Where F2 • When after scoring • How empirical quantiles
- Inputs: {CLV^{$}_{i,H}}
- Algebra: compute chosen percentiles (e.g., P90, P50)
- Output: q_{0.9}, q_{0.5}

**F4 [Component] — Tier assignment**
- Why segmentation • What VIP/High/Med/Low • Where F2–F3 • When final • How compare
- Inputs: CLV^{$}_{i,H}, q_{•}
- Algebra: VIP if ≥ q_{0.9}; High if ∈ [q_{0.5}, q_{0.9}); Med if (0, q_{0.5}); Low if = 0 (adjust scheme as needed)
- Output: tier label

---

## Compute Helpers (only for B4, C2, D3)

**Hypergeometric series (C2) — pseudo**
function hypergeom2F1(A,B,C,z, eps=1e-12, max_terms=200):
    u = 1.0
    s = 1.0
    for j in 1..max_terms:
        u *= ((A+j-1)*(B+j-1))/((C+j-1)*j) * z
        s += u
        if abs(u) < eps: break
    return s

**BG/NBD fit (B4) — objective & constraints**
Objective: maximize J(r,α,a,b) = Σ_i ln L_i with ln L_i as in Band B.
Constraints: r>0, α>0, a>0, b>0. Use multiple random positive starts.

**Gamma–Gamma fit (D3) — objective & constraints**
Objective: maximize K(p,q,γ) = Σ_{i:x_i>0} ln L_i^{GG}.
Constraints: p>0, q>1, γ>0. Use multiple random positive starts.

---

One-line dependency rail: A1→A2→A3/A4→A5 → B1→B2→B3→B4 → C1→C2→C3 → D1→D2→D3 → E1→E2 → F1→F2→F3→F4
