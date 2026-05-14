import json
import glob

REMOVE_KEYS = {
    'meta', 'class_list', 'newspack_spnsrs_tax', 'brand',
    'yoast_head', 'yoast_head_json', 'schema', 'parsely', '_links',
}

def strip_keys(obj):
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in REMOVE_KEYS}
    elif isinstance(obj, list):
        return [strip_keys(item) for item in obj]
    return obj

files = sorted(glob.glob('posts_*.json'), key=lambda f: int(f.split('_')[1].split('.')[0]))

all_posts = []
for f in files:
    with open(f) as fh:
        all_posts.extend(strip_keys(json.load(fh)))
    print(f"Loaded {f} ({len(all_posts)} total posts so far)")

with open('streetcarsuburbs.json', 'w') as fh:
    json.dump(all_posts, fh, indent=4)

print(f"Done. {len(all_posts)} posts written to streetcarsuburbs.json")
