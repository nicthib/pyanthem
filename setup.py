import setuptools

with open('README.rst', 'r') as fh:
	long_description = fh.read()

setuptools.setup(
	name='pyanthem',
	version='0.68',
	author='Nic Thibodeaux',
	author_email='dnt2111@columbia.edu',
	description='pyanthem - a neuroimaging audiovisualiation tool.',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/nicthib/pyanthem',
	packages=setuptools.find_packages(),
	setup_requires=[
		'numpy',
		'scipy'],
	install_requires=[
		'midiutil',
		'matplotlib',
		'pygame',
		'sklearn',
		'requests',
		'googledrivedownloader',
		'pillow',
		'Pmw'
	  ],
	include_package_data=True,
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
	],
	python_requires='>=3.7',
)
