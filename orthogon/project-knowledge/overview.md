# Step 0: Orthogonal Expert Training - Overview & Context

## **What We're Building**
Adding orthogonal expert training to your existing HGNN-MoE architecture to create specialized, non-redundant expert representations that automatically detect context degradation.

## **Why This Matters**
**Current State:** Your HGNN-MoE has experts that communicate densely via graph coupling, but experts may learn overlapping/redundant representations.

**Target State:** 
- **Orthogonal experts** = Each expert learns a unique, non-overlapping "cognitive direction"
- **No information redundancy** = Maximum information density per parameter
- **Emergent context awareness** = When input degrades, experts can't maintain orthogonality → automatic degradation detection
- **Foundation for hierarchical architecture** = Clean basis for the nested LoRA compression layers

## **How It Works Conceptually**
Think of experts as **basis vectors** in representation space:
- **Before:** Experts might learn similar patterns (like having 3 experts all focusing on syntax)
- **After:** Each expert becomes a unique "axis" (syntax, semantics, pragmatics, etc.)
- **Graph coupling** still handles dense communication, but now information flows between truly specialized experts
- **Orthogonality loss** encourages experts to maintain distinct specializations

## **Technical Implementation Strategy**
1. **Soft constraints first** - Add orthogonality loss to encourage expert differentiation
2. **Monitor and analyze** - Track how expert specializations evolve during training  
3. **Gradual strengthening** - Increase orthogonality constraints as training progresses
4. **Preserve existing architecture** - Your HGNN coupler stays exactly the same

## **Success Metrics**
- Experts develop measurably different attention patterns
- Maintained or improved perplexity scores
- Clear expert specialization visible in communication matrices
- Foundation ready for nested architecture expansion

## **Integration Points**
- **Minimal disruption** to your proven HGNN architecture
- **Additive changes** - orthogonality loss runs alongside existing training
- **Configurable** - can disable orthogonality constraints to compare performance

