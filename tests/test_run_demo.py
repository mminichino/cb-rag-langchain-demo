#!/usr/bin/env python
#
import re
import sys
import os
from streamlit.web import cli as stcli

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(current)

if __name__ == '__main__':
    parameters = sys.argv[1:]
    sys.argv = ["streamlit", "run", "cbragdemo/chat_with_pdf.py", "--"]
    sys.argv.extend(parameters)
    sys.exit(stcli.main())
