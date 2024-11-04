import pytest

def test_test(test_fixture):
    """
    Just to check if pytest is working as expected. Can be removed later in exchange for real tests.
    """
    assert sum(test_fixture)== 3