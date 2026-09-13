# Bounded algebraic diagnostic

Read-only with respect to benchmark inputs, previous sources/binaries/results,
verifier, and leaderboard. No distance search, logical-word enumeration, code
construction, witness generation, publication or GPU work.

Use the same six frozen component-study inputs. Analyze both CSS sectors:
1. Exact coordinate direct sums of ker(H) via RREF fundamental components,
   equivalent to masks H diag(m) G^T=0. Apply to whole checks and all documented
   coordinate subsets. Distinguish zero-kernel coordinates and nontrivial logical
   images. Logical rank on S is |S|-rank(Hopp_S)-rank(Hown)+rank(Hown_complement).
2. Candidate cut coupling rank r(H_S)+r(H_T)-r(H). Inspect every prefix of eight
   fixed orderings: natural, reverse, three breadth-first traversals of the RREF
   support graph, three seeded random orderings (1900..1902). Report balanced cuts
   with both sides >=1/4 of the domain; repeat inside every exact active component
   of size >=32. Also assess documented subsets. No claim of optimal separators.
3. Test simultaneous cyclic shifts within every equal consecutive block length
   L>=3 dividing n. Verify BOTH check row spaces, not adjacent-row scores. These
   are layout-based proposals, not arbitrary hidden-symmetry recovery. For accepted
   n/2 shifts, test an exact standard two-block circulant row-space representation
   from one X check and its swapped reciprocal Z generator. Only after that gate,
   compute binary polynomial gcd, Bezout certificates and complete factorization
   of x^L+1; compare predicted kernel/quantum dimensions to matrix ranks. CRT
   factors are algebraic subspaces, not independent physical supports.

Verify exact decompositions and accepted symmetries under random invertible row
mixing and the already frozen joint row/column permutations. Known permutations
are used only to audit equivariance, never claimed as inferred. Polynomial factors
must multiply back and pass independent irreducibility checks. Synthetic tests
compare decomposition with exhaustive mask equations and ranks with syndrome
intersection dimensions. Freeze source hashes before the real-input diagnostic.

Report algebraic opportunities, candidate-cut limits, timing and complete raw
metadata. A useful next search requires an actual new logically nontrivial small
coordinate space, a small measured balanced coupling rank, or a clearly identified
verified polynomial reduction. Algebraic factor size alone is not a distance
speedup. Stop after diagnosis; no adaptive distance-search campaign this turn.
