#!/usr/bin/env python
import sys
from init import installer_script
from graphdb_builder.builder import builder as builder
import debug

if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == 'init':
            installer_script()
        elif sys.argv[1] == 'debug':
            debug.main()
        else:
            builder.main()
