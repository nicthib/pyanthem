# conda remove --name pyanthem --all
import setuptools
with open('README.rst', 'r') as fh:
	long_description = fh.read()
setuptools.setup(
	name='pyanthem',
	version='1.1.7',
	author='Nic Thibodeaux',
	author_email='dnt2111@columbia.edu',
	description='pyanthem - an audiovisualization tool to make your data more interesting',
	long_description=long_description,
	long_description_content_type='text/x-rst',
	url='https://github.com/nicthib/pyanthem',
	packages=setuptools.find_packages(),
	setup_requires=[
		'numpy',
		'scipy'],
	install_requires=[
		'midiutil',
		'matplotlib',
		'pygame==2.0.0.dev10',
		'sklearn',
		'requests',
		'googledrivedownloader',
		'pillow',
		'mido',
		'ttkthemes',
		'h5py'
	  ],
	include_package_data=True,
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	python_requires='>=3.7',
)
