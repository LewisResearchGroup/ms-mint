from ms_mint.tools import generate_grid_peaklist


def test__generate_peaklist():
    peaklist = generate_grid_peaklist([115], .1, intensity_threshold=10000)
    assert peaklist is not None