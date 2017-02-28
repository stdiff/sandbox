try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Homemade Set',
    'author': 'Hironori Sakai',
    'url': 'https://github.com/stdiff/sandbox',
    'download_url': 'https://github.com/stdiff/sandbox',
    'author_email': 'hsakai@stdiff.net',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['Menge'],
    'scripts': [],
    'name': 'Menge'
}

setup(**config)
