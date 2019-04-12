import sys
import logging

logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/home/ubuntu/flaskproject/")

from dash_mess import server as application
