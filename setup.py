from setuptools import setup
import os

import versioneer


# Create list of data files
def find_data_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join("..", path, filename))

    return paths


extra_files = find_data_files("bb_astromodels/data")


setup(
    name="bb_astromodels",
    version="0.1",
    include_package_data=True,
    package_data={"": extra_files},
    packages=["bb_astromodels"],
)
