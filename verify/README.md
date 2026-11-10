# Verification supplement: two-letter orbits of the strong-force three-body problem

Everything here checks the claims independently of the research code in `tools/`: no file in this folder imports
from it. Requirements: Python 3 with numpy (and sympy for the threshold).

## The claims
Planar three-body problem, potential U = sum m_i m_j / r_ij^2, energy 0, angular momentum 0.
1. For masses with m_k^2 (m_i + m_j)^3 < m_i^2 m_j^2 M (M the total mass), the two-letter syzygy class (ij) is
   realised by a periodic orbit (periodic up to a rotation, i.e. closed on the shape sphere). For masses (1, 1, m3)
   the condition is m3 < (1 + sqrt 65) / 16 = 0.5663911...
2. The condition is the limit at the collision of the sign function of Montgomery's curvature formula (Montgomery
   2005, eqs. (7.12)-(7.13) of the arXiv version), equivalently: a loop round the ij collision shorter than the
   asymptotic cylinder exists (a "waist").
3. Without a waist at any end, the class (12) has no closed orbit: reduced in the paper (Proposition G) to three
   conditions; (I) is proved here by interval arithmetic (five triples and the family (1, 1, m), 0.567 <= m <= 1.07),
   (II) and (III) are checked numerically at five triples.

## Files
- `certificates/orbits_12.json`: nine orbits in the class (12) (masses (1, 1, m3) for m3 = 0.4, 0.5, 0.52, ...,
  0.56, and (1, 0.5, 0.30), (1, 0.2, 0.15)): initial positions and velocities to full double precision, the
  period, the action and the syzygy word.
- `verify_orbits.py`: for each certificate checks that the momentum, angular momentum and energy vanish at the
  start; integrates one period with its own fixed-step Runge-Kutta 4 (two step sizes, Richardson); checks closure
  on the shape sphere (mutual distances and their rates), the action, and that the syzygy sequence over the period
  is exactly [1, 2].
- `verify_search.py`: the reversible search of the paper's Section 5 at nineteen mass triples (2000 starts each,
  fixed-step RK4 in a regularised time at two step sizes; every sign change of the shooting function refined to tell
  a root from a jump between branches): a root at the start parameter of each of the nine certified orbits, none at
  the ten others (0.57 to 0.75, (1, 0.5, 0.45), (1, 0.2, 0.19) and two controls without a waist at the 12-end). It sees
  only orbits with the reversing symmetry, and it compares neighbouring starts that both reach a body-2 syzygy; a sign
  change across a gap of other starts (bounded by collision orbits) is not examined.
- `verify_stability.py`: the trace and rotation columns of Table 1: the normal Jacobi equation along each certified
  orbit (two step sizes), the determinant of the return map, the rotation per period, and the rotation again as the
  geometric phase (half the solid angle the shape curve encloses about C12).
- `verify_birth.py`: the birth law in high precision (mpmath): A4(m*), the constant C = 35.12, the ratios for the
  shortest round loops, and for all nine certified orbits the action below the shortest round loop of the class.
- `verify_threshold.py`: (A) symbolically, that Montgomery's criterion at the 12 collision is equivalent to the
  mass condition and gives 8 m^2 - m - 2 = 0 for masses (1, 1, m); (B) numerically, in configuration space with no
  shape-sphere coordinates, that a loop round the 12 collision is shorter than the cylinder's circumference exactly
  when the mass condition holds (14 mass sets, including 0.566 / 0.567 on both sides of the threshold).
- `verify_kappa2_rational.py`: exact rational expansions of the curvature at the collision (slow).
- `verify_I_interval.py`: a proof of condition (I) at the five mass triples by interval arithmetic (outward-rounded
  IEEE +, -, *, /, sqrt only; masses and constants as intervals): in stereographic charts centred at the collisions,
  K < 0 iff Q = F Lap F - |grad F|^2 > 0 for a rational F smooth at the collision; Q is enclosed with second-order
  interval jets over adaptively refined boxes covering each collision's Voronoi cell outside sigma_0. Then a negative
  control (a tiny loop in place of sigma_0) that must fail. The inside-sigma_0 test of a box uses floating point
  with a margin. `cover_slice` in the same file proves K < 0 outside one loop on whole mass intervals: with the loop
  sigma_0 of (1, 1, 0.57), for all (1, 1, m3) with 0.567 <= m3 <= 2 (365 mass intervals; log in
  `verify_I_interval_slice_output.txt`). That loop is a valid sigma_0 (convex for each mass, enclosing C12 only, clear
  of arc 3) for m3 <= 1.077, so condition (I) holds on [0.567, 1.07].
- `certificates/chains.npz`, `certificates/chains_meta.json`: for five mass sets without a waist ((1, 1, 0.57),
  (1, 1, 0.6), (1, 1, 2), (1, 0.5, 0.45), (1, 0.2, 0.19)) chains of 261, 153, 43, 35 and 659 nested convex loops
  round the 12 collision (curve-shortening leaves), as points of a stereographic chart.
- `verify_converse.py`: the converse's reduction (Proposition G) at those five mass sets, with its own curvature
  formula, charts and fixed-step RK4 geodesic integrator (every integration repeated with half the step): no waist at
  any end; U by two routes; Lemma K against its closed form; (I) K < 0 outside the outer loop on a 400 x 800 grid,
  and the outer loop clear of the equator arc 3; (II) every loop convex, the loops nested, and for each loop one time
  T_k such that all inward normals keep the Jacobi field G > 0, G' < 0 on [0, T_k], the parallels are simple and
  nested, and the T_k-parallel lies inside the next loop; condition C out to the last loop; (III) the Fermi normals of
  the three equator arcs, including 30 from each collision cap down to rho = 1e-3, keep G > 0, G' > 0 and leave the
  hemisphere in monotone order (normals that enter another collision's cap are followed until they cross the
  equator there or go down the end too steeply to turn back; none returns to the hemisphere); both step sizes must
  pass; Pi_H < pi. Then the negative control (1, 1, 0.5), which has a waist: the no-waist
  check and (III) on the two arcs at C12 must fail, and do.

Saved outputs: `verify_orbits_output.txt`, `verify_threshold_output.txt`, `verify_search_output.txt`,
`verify_converse_output.txt`, `verify_stability_output.txt`, `verify_birth_output.txt`,
`verify_kappa2_rational_output.txt`, `verify_I_interval_output.txt`,
`verify_I_interval_slice_output.txt`.

## Run
    python verify/verify_threshold.py      # seconds
    python verify/verify_orbits.py         # some minutes (2**17 and 2**18 RK4 steps per orbit)
    python verify/verify_search.py         # about fifteen minutes
    python verify/verify_stability.py      # some minutes
    python verify/verify_birth.py          # some minutes
    python verify/verify_I_interval.py     # about a minute
    python verify/verify_converse.py       # about twenty minutes (or one mass set: verify_converse.py 1,1,0.57)
Each prints its table and ends with ALL PASS or SOME FAILED.

## Status of the statement
Proved: claim 1 in the direction "waist => orbit" (and the orbits are checked here). Known: no two-letter orbit
for equal masses (Montgomery 2005). The converse for unequal masses without a waist follows from Proposition G once
its three conditions hold; `verify_converse.py` checks all three at five mass sets (ALL PASS).

## What this does not check
Condition (I) is proved (verify_I_interval.py). The other converse checks are numerical (double precision, fixed-step integration, sampled normals and grids), not
interval arithmetic, and they cover five mass sets, not all masses without a waist. Below rho = 1e-3 at the collisions
the normals are not integrated; the paper relies there on the near-cylinder form of the ends. The band just above the
threshold, where every margin vanishes with the curvature coefficient, needs a separate argument. The search sees only
symmetric orbits.
