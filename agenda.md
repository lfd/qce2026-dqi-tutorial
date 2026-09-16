# Agenda
The tutorial is set on Tuesday, September 15, and spans two sessions (10:00 - 11:30 and 1:00 - 2:30).
The first session builds the classical coding-theory foundation that DQI rests on - linear codes and syndrome decoding, Reed-Solomon codes, and LDPC codes - and then presents the DQI algorithm itself.
The second session shows what DQI can do on structured and on sparse problems and closes with a hands-on part on encoding practical problems in DQI-native formats.
Hands-on exercises are interleaved with the slides; they live in `exercises.ipynb`, whose four parts follow the four blocks below.

## Session 1 (10:00 AM - 11:30 AM) - Coding theory and the DQI algorithm
- **10:00 - 10:08:** Introduction and motivation: decoding as optimisation, tutorial roadmap, the hands-on environment
- **10:08 - 10:28:** Coding theory: error-correcting codes, the Hamming code, Hamming and minimum distance, generator and parity-check matrices, syndrome decoding
  - *Exercise 1:* Decoding the Hamming code
- **10:28 - 10:43:** Reed-Solomon codes: polynomials over finite fields, RS generator matrices, Berlekamp-Welch decoding
  - *Exercise 2:* Decoding Reed-Solomon codes
- **10:43 - 10:55:** LDPC codes: worst case vs. average case, Shannon's noisy-channel coding theorem, Gallager's ensemble, Tanner graphs, bit flipping and belief propagation
- **10:55 - 11:30:** The DQI algorithm: Max-LINSAT, the DQI objective function, the interference idea, $f(\mathbf{x})^k$ and the reduction to syndrome decoding, algorithm steps, general polynomials, arbitrary $\mathbf{v}$, Max-LINSAT with set constraints and the phase oracle
  - *Exercise 3:* Max-Cut with DQI
  - *Exercise 4:* 3-COLOR with DQI

## Session 2 (1:00 PM - 2:30 PM) - What DQI can do and problem encodings
- **1:00 - 1:25:** DQI on structured problems: approximation ratio, the semi-circle law, DQI with Reed-Solomon decoders, Optimal Polynomial Intersection and its classical competitors
  - *Exercise 5:* Optimal Polynomial Intersection
- **1:25 - 1:45:** Sparse Max-XORSAT and the current state of research: sparse Max-XORSAT as LDPC decoding, DQI with imperfect decoders, 3-Max-XORSAT results, where that leaves DQI
- **1:45 - 2:20:** Encoding problems for DQI with DQI-Kit: DQI-Kit architecture and features, QUBO vs. Max-LINSAT, linear (in)equalities, objectives as weighted constraints, Boolean constraints, linear dependencies and anchoring
  - *Exercise 6:* Production planning
  - *Exercise 7:* Boolean constraints
  - *Exercise 8:* Shift planning
  - *Exercise 9:* Dependency reduction
- **2:20 - 2:30:** What we learned and what is next + time buffer
