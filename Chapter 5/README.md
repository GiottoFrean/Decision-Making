# Structure Learning

Beyond learning parameters, we can learn the **structure** of the graphical model itself - which variables depend on which others.

---

## 1 - Searching PGMs

This notebook addresses the question: given data, what is the best graph structure?

The challenge is that the space of possible graphs grows super-exponentially with the number of variables. For just 3 variables, there are many possible structures:
- A → B → C
- A → B ← C  
- A ← B → C
- A ← B ← C
- And many more with different edge directions

**Approach:**
1. Define a **scoring function** (e.g., likelihood, BIC, or AIC) that measures how well a structure fits the data
2. **Search** through possible structures to find the best score
3. Use heuristics like greedy search, hill climbing, or more sophisticated algorithms

**Key insight:** Multiple graphs can represent the same set of conditional independencies. These form **equivalence classes**. For example:
- A → B → C
- A ← B ← C  
- A ← B → C

All three encode: "A and C are conditionally independent given B." We can't distinguish between them from data alone - only their independence structure matters.

The notebook demonstrates this by testing different 3-variable structures and comparing their likelihoods on sample data. With enough data, the true structure (or its equivalence class) should score highest.

**Challenges:**
- Computational cost of searching exponentially many structures
- Need sufficient data to distinguish structures reliably
- Regularization (via BIC/AIC penalties) to avoid overfitting with complex graphs

Structure learning is powerful but difficult. It's most practical when:
- You have domain knowledge to constrain the search
- The number of variables is modest
- You have large amounts of data
