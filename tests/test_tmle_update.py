import pytest
from pytmle.tmle_update import tmle_update


def test_tmle_update_not_implemented(mock_tmle_update_inputs):
    with pytest.raises(NotImplementedError):
        # TODO: Adapt as soon as tmle_update has been implemented
        tmle_update(**mock_tmle_update_inputs)
