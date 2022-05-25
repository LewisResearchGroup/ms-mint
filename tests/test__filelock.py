import os
from ms_mint.tools import lock


def test_create_file(tmpdir):
    # This test does not work as expected
    p = tmpdir.mkdir("sub").join("test.txt")
    p.write("content")
    with lock(p):
        p.write("content")
        print(os.access(p, os.W_OK))
        # This should fail
        # will have to find a better way
        # to test this.
