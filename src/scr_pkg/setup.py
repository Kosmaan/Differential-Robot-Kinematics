from setuptools import find_packages, setup

package_name = 'scr_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Iancu Andrei Iulian',
    maintainer_email='andrei-iulian.iancu@academic.tuiasi.ro',
    description='Package created to control the novnc gazebo simmulation made for educational purposes',
    license='Free software',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scr_node = scr_pkg.scr:main'
        ],
    },
)
