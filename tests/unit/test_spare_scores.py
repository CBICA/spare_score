import pytest
import spare_scores
import re

def test_spare_test():

    # test no arguments given:
    no_args = "spare_test() missing 2 required positional " + \
                 "arguments: 'df' and 'mdl_path'"
    with pytest.raises(TypeError, match=re.escape(no_args)):
        spare_scores.spare_test()

    
def test_spare_train():

    pass

def test_expspace():
    pass

def test_load_df():
    pass