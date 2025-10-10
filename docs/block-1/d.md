
# Model 1 — CLV via BG/NBD + Gamma–Gamma (Math Markdown)

> **Render tips for VS Code**
> - Use the **Markdown Preview Enhanced** extension (or any KaTeX/MathJax-enabled preview).
> - Inline math: `$ ... $`; Display math: `$$ ... $$`.
> - Units: time in **days**, money in **cents**.

---

## Band A — Foundational Quantities (from orders)

Let a customer \(i\) have order times \(t_{i1} \le t_{i2} \le \dots \le t_{in_i} \le t_{\mathrm{cal}}\) (all in one timezone, e.g., UTC) and per-order **net** margins \(m_{ik}\) (cents).

**Counts & repeats**
$$
n_i = \bigl|\{k: t_{ik} \le t_{\mathrm{cal}}\}\bigr|, 
\qquad
x_i = \max(n_i - 1,\; 0).
$$

**Ages (days)**
$$
T_i =
\begin{cases}
\bigl(t_{\mathrm{cal}} - t_{i1}\bigr)_{\text{days}}, & n_i \ge 1,\\[4pt]
0, & n_i = 0,
\end{cases}
\qquad
t_{x,i} =
\begin{cases}
\bigl(t_{in_i} - t_{i1}\bigr)_{\text{days}}, & n_i \ge 1,\\[4pt]
0, & n_i = 0.
\end{cases}
$$

**Per-order net margin and mean margin**
$$
m_{ik} \;=\; p_{ik}\,q_{ik} \;-\; d_{ik} \;-\; c_{ik} \;-\; s_{ik} \;-\; \tau_{ik} \;-\; r_{ik},
$$
where \(p_{ik}\) = unit price (cents), \(q_{ik}\)=quantity, \(d_{ik}\)=discount (cents),
\(c_{ik}\)=COGS (cents), \(s_{ik}\)=shipping (cents), \(\tau_{ik}\)=tax (cents), \(r_{ik}\)=refunds (cents).

$$
\bar z_i \;=\;
\begin{cases}
\dfrac{1}{n_i}\displaystyle\sum_{k=1}^{n_i} m_{ik}, & n_i>0,\\[10pt]
0, & n_i=0.
\end{cases}
$$

---

## Band B — BG/NBD Parameter Fit

**Log-Beta identity (to avoid special functions)**
$$
\ln B(u,v)=\ln\Gamma(u)+\ln\Gamma(v)-\ln\Gamma(u+v).
$$

For a trial parameter vector \((r,\alpha,a,b)\) with all components \(>0\) and one customer’s
\((x_i,T_i,t_{x,i})\), define the four log-pieces:
$$
\begin{aligned}
\ln A^{(i)}_1 &= \ln B(a,\,b+x_i)-\ln B(a,b),\\[2pt]
\ln A^{(i)}_2 &= \ln\Gamma(r+x_i)-\ln\Gamma(r) + r\ln\alpha,\\[2pt]
\ln A^{(i)}_3 &= -(r+x_i)\,\ln(\alpha+T_i),\\[2pt]
\ln A^{(i)}_4 &= \mathbf{1}_{\{x_i>0\}}\!\Big(\ln B(a+1,\,b+x_i-1)-\ln B(a,b) -(r+x_i)\ln(\alpha+t_{x,i})\Big).
\end{aligned}
$$

**Per-customer log-likelihood (log-sum-exp form)**
$$
\ln \mathcal{L}_i
=\ln\!\Big(\exp(\ln A^{(i)}_1+\ln A^{(i)}_2+\ln A^{(i)}_3)\;+\;\exp(\ln A^{(i)}_2+\ln A^{(i)}_4)\Big).
$$

**Total objective and estimate**
$$
\mathcal{J}(r,\alpha,a,b)=\sum_i \ln \mathcal{L}_i,
\qquad
(\hat r,\hat\alpha,\hat a,\hat b)=\arg\max_{r,\alpha,a,b>0}\;\mathcal{J}(r,\alpha,a,b).
$$

---

## Band C — BG/NBD Horizon Expectation \(E[N_H]\)

Fix a forecast horizon \(H>0\) (days). For each customer \(i\), set
$$
A_i=\hat r+x_i,\qquad
B_i=\hat b+x_i,\qquad
C_i=\hat a+\hat b+x_i-1,
$$
$$
z_i=\frac{H}{\hat\alpha+T_i+H}\in(0,1),\qquad
\text{pow}_i=\Big(\frac{\hat\alpha+T_i}{\hat\alpha+T_i+H}\Big)^{A_i}.
$$

**Gaussian hypergeometric function** (power series, Pochhammer symbol \((q)_j\))
$$
{}_2F_1(a,b;c;z)=\sum_{j=0}^{\infty} \frac{(a)_j\,(b)_j}{(c)_j}\,\frac{z^j}{j!},
\qquad
(q)_0=1,\;\; (q)_j=q(q+1)\cdots(q+j-1).
$$

**Practical recurrence for computation (finite \(J\))**
Let \(u_0=1\) and, for \(j\ge 1\),
$$
u_j = u_{j-1}\cdot \frac{(A_i+j-1)(B_i+j-1)}{(C_i+j-1)\,j}\cdot z_i,
\qquad
S_i=\sum_{j=0}^{J} u_j \;\approx\; {}_2F_1(A_i,B_i;C_i;z_i).
$$

**Finite-horizon expected transactions**
$$
E_i[N_H]
=\frac{\hat a+\hat b+x_i-1}{\hat a-1}\Big(1-\text{pow}_i\,S_i\Big)
+\mathbf{1}_{\{x_i>0\}}\;\frac{\hat a}{\hat b+x_i-1}
\Big(\frac{\hat\alpha+T_i}{\hat\alpha+t_{x,i}}\Big)^{A_i}.
$$

---

## Band D — Gamma–Gamma Parameter Fit

Using only customers with (x_i>0) and observed mean margin (\bar{z}_i>0), for trial ((p,q,\gamma)) with (p>0,\ q>1,\ \gamma>0) (natural logs):

```math
\ln \mathcal{L}^{\mathrm{GG}}_i
= \ln\Gamma(p x_i + q) - \ln\Gamma(p x_i) - \ln\Gamma(q)
+ (p x_i - 1)\,\ln \bar{z}_i + (p x_i)\,\ln x_i
+ q\,\ln\gamma - (p x_i + q)\,\ln(\gamma + x_i \bar{z}_i)
```

**Total objective and estimate**

```math
\mathcal{K}(p,q,\gamma)
= \sum_{i:\, x_i>0} \ln \mathcal{L}^{\mathrm{GG}}_i,
\qquad
(\hat{p}, \hat{q}, \hat{\gamma})
= \operatorname*{arg\,max}_{p>0,\, q>1,\, \gamma>0}\ \mathcal{K}(p,q,\gamma)
```

---

## Band E — Expected Margin per Transaction

Given ((\hat p,\hat q,\hat\gamma)) and a customer’s (x_i,\bar z_i),
$$
E_i[M]=\frac{\hat p,(\hat\gamma + x_i,\bar z_i)}{\hat p,x_i + \hat q - 1}\quad\text{(cents per transaction)}.
$$


---

## Band F — CLV and Tiers

**Horizon CLV (cents)**
$$
\text{CLV}_{i,H} \;=\; E_i[N_H]\cdot E_i[M].
$$

**Optional dollars**
$$
\text{CLV}^{\$}_{i,H}=\frac{\text{CLV}_{i,H}}{100}.
$$

**Example tiering** (using empirical percentiles \(q_{0.9},q_{0.5}\) on the chosen CLV scale)
- VIP if \(\text{CLV}\ge q_{0.9}\)  
- High if \(q_{0.5}\le \text{CLV}< q_{0.9}\)  
- Med if \(0<\text{CLV}<q_{0.5}\)  
- Low if \(\text{CLV}=0\)

---

## Variable Glossary

- \(t_{\mathrm{cal}}\): calibration end timestamp (fixed).  
- \(n_i\): number of observed orders in calibration for customer \(i\).  
- \(x_i\): repeat count \(=\max(n_i-1,0)\).  
- \(T_i\): time from first purchase to \(t_{\mathrm{cal}}\) (days).  
- \(t_{x,i}\): time from first to last purchase (days).  
- \(m_{ik}\): per-order net margin (cents); \(\bar z_i\) is its mean per customer.  
- \(r,\alpha,a,b\): BG/NBD parameters; hats denote MLEs.  
- \(p,q,\gamma\): Gamma–Gamma parameters; hats denote MLEs.  
- \(H\): forecast horizon in days.  
- \(E_i[N_H]\): expected future transactions over \(H\).  
- \(E_i[M]\): expected net margin per transaction (cents).  
- \(\text{CLV}_{i,H}\): horizon CLV (cents).

