import json
import pprint
from utils import fix_incomplete_json

result = json.loads('{\"group_by_col\": \"生产年份\", \"agg_col\": \"生产年份\", \"agg_func\": \"count\"}')
pprint.pprint(result)

str = json.loads(fix_incomplete_json('{\"group_by_col\": \"生产年份\", \"agg_col\": \"生产年份\", \"agg_func\": \"count\"}'))

pprint.pprint(str)