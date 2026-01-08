import stimdx
from stimdx import context as ctx
import pytest

def test_context_proxy_rec():
    c = stimdx.Circuit()
    c.block("H 0\nM 0")
    
    # Use ctx.rec(-1) instead of lambda
    c.let("m0", ctx.rec(-1))
    
    # Use ctx.vars["m0"]
    c.emit(ctx.vars["m0"], name="out_m0")
    
    sampler = c.compile_sampler(seed=123)
    results = sampler.sample_with_classical(shots=10)
    
    for res in results:
        meas = res["measurements"][-1]
        out = res["outputs"][0]
        assert out == meas
        assert res["vars"]["m0"] == meas

def test_context_proxy_vars_chaining():
    # Test assigning one var to another using the proxy
    c = stimdx.Circuit()
    c.block("H 0\nM 0")
    
    c.let("a", ctx.rec(-1))
    c.let("b", ctx.vars["a"])
    c.emit(ctx.vars["b"], name="b_out")
    
    sampler = c.compile_sampler(seed=42)
    results = sampler.sample_with_classical(shots=5)
    
    for res in results:
        assert res["vars"]["a"] == res["measurements"][-1]
        assert res["vars"]["b"] == res["vars"]["a"]
        assert res["outputs"][0] == res["vars"]["b"]
