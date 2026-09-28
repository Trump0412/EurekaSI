import pytest
from spatial_intelligence.torch_numeric_contract import capture_numeric_contract, restore_numeric_contract


def test_numeric_policy_roundtrip():
    original=capture_numeric_contract()
    try:
        altered={k:not v for k,v in original.items()}
        restore_numeric_contract(altered)
        assert capture_numeric_contract()==altered
        restore_numeric_contract(original)
        assert capture_numeric_contract()==original
    finally:
        restore_numeric_contract(original)


def test_partial_contract_rejected_without_mutation():
    original=capture_numeric_contract()
    with pytest.raises(ValueError):restore_numeric_contract({'matmul_allow_tf32':True})
    assert capture_numeric_contract()==original
