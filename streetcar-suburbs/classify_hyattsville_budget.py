import json
import re
import argparse
import llm

def strip_html(html):
    return re.sub(r'<[^>]+>', '', html or '').strip()

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=None, help='Process only the first N stories (for testing)')
args = parser.parse_args()

model = llm.get_model("qwen3:4b")

with open('streetcarsuburbs.json') as f:
    stories = json.load(f)

if args.limit:
    stories = stories[:args.limit]

matches = []

for i, story in enumerate(stories):
    title = strip_html(story.get('title', {}).get('rendered', ''))
    excerpt = strip_html(story.get('excerpt', {}).get('rendered', ''))
    content = strip_html(story.get('content', {}).get('rendered', ''))

    text = f"Title: {title}\n\nExcerpt: {excerpt}\n\nContent (first 1000 chars): {content[:1000]}"

    prompt = (
        "You are a news classifier. Does the following story primarily cover the city of Hyattsville's "
        "municipal budget (e.g., city budget proposals, budget hearings, fiscal year spending, budget amendments, "
        "tax rates set by Hyattsville city government)? "
        "Reply with only YES or NO.\n\n"
        f"{text}"
    )

    response = model.prompt(prompt, stream=False)
    answer = response.text().strip().upper()
    is_match = answer.startswith("YES")

    print(f"[{i+1}/{len(stories)}] {title[:80]} -> {answer}")

    if is_match:
        matches.append(story)

with open('hyattsville_budget.json', 'w') as f:
    json.dump(matches, f, indent=4)

print(f"\nDone. {len(matches)} matching stories written to hyattsville_budget.json")
