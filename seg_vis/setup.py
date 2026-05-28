from setuptools import setup

package_name = 'seg_vis'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='selina-xiangqi',
    maintainer_email='2107931860@qq.com',
    description='Segmentation visualization helper',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_colorizer = seg_vis.segmentation_colorizer:main',
        ],
    },
)
