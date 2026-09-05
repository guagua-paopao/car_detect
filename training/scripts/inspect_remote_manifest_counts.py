import csv, collections, sys
for p in sys.argv[1:]:
    with open(p, encoding='utf-8-sig', newline='') as f: rows=list(csv.DictReader(f))
    print('\n', p, len(rows), list(rows[0]) if rows else [])
    for split in ('train','validation','test'):
        rs=[r for r in rows if r.get('split')==split]
        sup=sum((r.get('color','').lower() not in {'','unknown'} and r.get('color_supervised','true').lower() not in {'false','0','no'}) for r in rs)
        print(split, len(rs), 'color_sup', sup, 'sources', collections.Counter(r.get('source_dataset') for r in rs).most_common())
    print('sample', {k: rows[0].get(k) for k in rows[0]} if rows else {})
