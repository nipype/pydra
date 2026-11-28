import os

from pydra.conftest import IMAGE_CACHE_VARS


def test_each_test_process_pulls_into_its_own_image_cache(private_image_cache):
    for var in IMAGE_CACHE_VARS:
        assert os.environ[var] == str(private_image_cache)
    assert private_image_cache.is_dir()
