try: from setuptools import setup
except: from distutils.core import setup

setup(	long_description="Transformio creates, applies, and evaluates coordinate transformations for vector and raster geospatial data.", #open("README.rst").read(), 
	name="""transformio""",
	license="""MIT""",
	author="""Karim Bahgat""",
	author_email="""karim.bahgat.norway@gmail.com""",
	version="""0.1.0""",
	keywords="""transformio""",
	packages=['transformio'],
    install_requires=[
		'numpy==1.*'
	],
	classifiers=['License :: OSI Approved', 'Programming Language :: Python', 'Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'Intended Audience :: End Users/Desktop'],
	description="""Transformio creates, applies, and evaluates coordinate transformations for vector and raster geospatial data.""",
	)
