# Structure Learning

Beyond learning parameters, we can learn the **structure** of the graphical model itself - which variables depend on which others.

---

## 1 - Searching PGMs

This notebook addresses the question: given data, what is the best graph structure?

The challenge is that the space of possible graphs grows super-exponentially with the number of variables. The approach is to define a **scoring function** (e.g., likelihood, BIC, or AIC) that measures how well a structure fits the data, then search through possible structures.

**Key insight:** Multiple graphs can represent the same set of conditional independencies. These form **equivalence classes**. We can't distinguish between them from data alone - only their independence structure matters.

The notebook demonstrates this by testing different 3-variable structures and comparing their likelihoods on sample data. With enough data, the true structure (or its equivalence class) should score highest.

**Challenges:**
- Computational cost of searching exponentially many structures
- Need sufficient data to distinguish structures reliably
- Regularization (via BIC/AIC penalties) to avoid overfitting with complex graphs
