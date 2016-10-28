
words = []
while True:
    try:
        words.append(raw_input())
    except EOFError:
        break

cleaned = set([w.lower() for w in words if "'s" not in w and w.isalpha()])

print "\n".join(sorted(cleaned))
