import sys

from .bmcs_exceptions import BmCS_Exception
from .BmCS import main

try:
    main()
    print("Done.")
    sys.exit(0)
except BmCS_Exception as be:
    print("BmCS exception: " + str(be))
    sys.exit(1)
except Exception as e:
    print("General exception: " + str(e))
    sys.exit(1)
