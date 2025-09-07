import json
import pprint
from utils import fix_incomplete_json

str = json.loads(fix_incomplete_json('{"filters": {"生产批次号": {"==": "BATCH20040409-472"}'))

pprint.pprint(str)