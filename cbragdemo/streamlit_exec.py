##
##

import sys
import os
from streamlit.web import cli as stcli

current = os.path.dirname(os.path.realpath(__file__))


def main():
    sys.argv = ["streamlit", "run", f"{current}/chat_with_pdf.py"]
    sys.exit(stcli.main())


if __name__ == '__main__':
    main()
