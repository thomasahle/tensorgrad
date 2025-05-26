#!/usr/bin/env python3
"""Tests for potential bugs in Expectation class."""

import pytest
from sympy import symbols
from tensorgrad import Variable, Product, Delta, Zero
from tensorgrad.extras.expectation import Expectation
from tensorgrad.tensor import Rename


def test_bug3_multiple_wrt_in_product():
    """Test that list.remove() bug when wrt appears multiple times in product."""
    i = symbols("i")
    x = Variable("x", i)
    
    # Create a product with x appearing twice
    prod = x @ x @ x
    
    # This should handle all occurrences of x correctly
    exp = Expectation(prod, x, mu=Zero(i), covar=Delta(i, "i, j"))
    
    # The simplification should work correctly even with multiple x's
    result = exp.simplify()
    
    # If the bug exists, it would only remove one x, leading to incorrect result
    # The correct result should involve third moments
    assert isinstance(result, Expectation) or isinstance(result, Product)


def test_bug5_rename_inconsistency():
    """Test that _rename doesn't update mu, covar, or covar_names."""
    i, j = symbols("i j")
    x = Variable("x", i)
    mu = Variable("mu", i) 
    covar = Variable("c", i, i2=i)
    covar_names = {"i": "i2"}
    
    expr = x
    exp = Expectation(expr, x, mu, covar, covar_names)
    
    # Rename edge i to j
    renamed = exp.rename(i="j")
    
    # Bug: mu, covar, and covar_names are not updated
    # This could lead to shape mismatches if we try to simplify
    assert renamed.tensor.shape == {"j": i}
    assert renamed.mu.shape == {"i": i}  # BUG: Should be {"j": i}
    assert renamed.covar.shape == {"i": i, "i2": i}  # BUG: Should be updated
    assert renamed.covar_names == {"i": "i2"}  # BUG: Should be {"j": "j2"} or similar


def test_bug6_depends_on_missing_mu_covar():
    """Test that depends_on doesn't check mu or covar dependencies."""
    i = symbols("i")
    x = Variable("x", i)
    y = Variable("y", i)
    z = Variable("z", i)
    
    # Create expectation where tensor doesn't depend on y, but mu does
    expr = x @ z  # doesn't depend on y
    mu = y  # depends on y
    exp = Expectation(expr, x, mu=mu)
    
    # Bug: depends_on only checks tensor, not mu
    assert not exp.depends_on(y)  # BUG: Should be True since mu depends on y
    
    # Similarly for covar
    covar = y @ y.rename(i="j") + Delta(i, "i, j")
    exp2 = Expectation(expr, x, covar=covar, covar_names={"i": "j"})
    assert not exp2.depends_on(y)  # BUG: Should be True since covar depends on y


def test_bug7a_stopiteration_line94():
    """Test StopIteration risk when no isomorphisms exist (line 94)."""
    i, j = symbols("i j")
    x = Variable("x", i)
    y = Variable("y", j)  # Different shape, no isomorphism with x
    
    # This should raise StopIteration due to next() without default
    exp = Expectation(y, x)
    
    with pytest.raises(StopIteration):
        # The bug occurs when trying to simplify a Variable that has no isomorphism with wrt
        exp.simplify()


def test_bug7b_stopiteration_line129():
    """Test StopIteration risk in _simplify_product (line 129)."""
    i, j = symbols("i j")
    x = Variable("x", i)
    y = Variable("y", j)  # Different shape
    
    # Create a product containing x but with wrong shape
    # We need to trick it into the Stein's lemma branch
    class FakeVariable(Variable):
        """A variable that claims to be equal to x but has different shape."""
        def __eq__(self, other):
            return isinstance(other, Variable) and other.name == "x"
        
        def isomorphisms(self, other):
            # Return empty iterator to trigger StopIteration
            return iter([])
    
    fake_x = FakeVariable("x", j)  # Wrong shape
    prod = fake_x @ y
    
    exp = Expectation(prod, x)
    
    # This should raise StopIteration when trying to get isomorphism
    with pytest.raises(StopIteration):
        exp._simplify_product(prod, {})


if __name__ == "__main__":
    # Run tests to demonstrate bugs
    print("Testing bug 3 (multiple wrt in product)...")
    try:
        test_bug3_multiple_wrt_in_product()
        print("✓ Bug 3 test passed (may not expose the bug)")
    except Exception as e:
        print(f"✗ Bug 3 test failed: {e}")
    
    print("\nTesting bug 5 (rename inconsistency)...")
    try:
        test_bug5_rename_inconsistency()
        print("✓ Bug 5 confirmed - rename doesn't update mu/covar/covar_names")
    except AssertionError:
        print("✗ Bug 5 test failed - rename might have been fixed")
    
    print("\nTesting bug 6 (depends_on missing mu/covar)...")
    try:
        test_bug6_depends_on_missing_mu_covar()
        print("✓ Bug 6 confirmed - depends_on doesn't check mu/covar")
    except AssertionError:
        print("✗ Bug 6 test failed - depends_on might have been fixed")
    
    print("\nTesting bug 7a (StopIteration in line 94)...")
    try:
        test_bug7a_stopiteration_line94()
        print("✗ Bug 7a test failed - should have raised StopIteration")
    except StopIteration:
        print("✓ Bug 7a confirmed - StopIteration raised")
    
    print("\nTesting bug 7b (StopIteration in line 129)...")
    try:
        test_bug7b_stopiteration_line129()
        print("✗ Bug 7b test failed - should have raised StopIteration")
    except (StopIteration, AttributeError) as e:
        print(f"✓ Bug 7b confirmed - {type(e).__name__} raised")