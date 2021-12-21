try: from setuptools import setup
except: from distutils.core import setup

requirements = [line.replace('\n','') for line in open('requirements.txt','r')]
requirements = [line for line in requirements
				if line and not line.startswith('#')]

setup(	long_description="Transformio creates, applies, and evaluates coordinate transformations for vector and raster geospatial data.", #open("README.rst").read(), 
	name="""transformio""",
	license="""MIT""",
	author="""Karim Bahgat""",
	author_email="""karim.bahgat.norway@gmail.com""",
	version="""0.1.0""",
	keywords="""transformio""",
	packages=['transformio'],
    install_requires=requirements,
	classifiers=['License :: OSI Approved', 'Programming Language :: Python', 'Development Status :: 4 - Beta', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'Intended Audience :: End Users/Desktop'],
	description="""Transformio creates, applies, and evaluates coordinate transformations for vector and raster geospatial data.""",
	)
