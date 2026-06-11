from setuptools import find_packages, setup
import os
from glob import glob

package_name = "depth_anything_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Orlando De Leon",
    maintainer_email="orlando.deleon@txdot.gov",
    description="Depth Anything V2 metric depth inference node for Pathfinder",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "depth_anything_node = depth_anything_node.node:main",
        ],
    },
)
