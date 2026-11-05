"""Shape invariant used by Bailando convolutional networks."""
def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"
