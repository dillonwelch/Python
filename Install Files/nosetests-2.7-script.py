#!C:\Python27\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'nose==1.1.2','console_scripts','nosetests-2.7'
__requires__ = 'nose==1.1.2'
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.exit(
        load_entry_point('nose==1.1.2', 'console_scripts', 'nosetests-2.7')()
    )
