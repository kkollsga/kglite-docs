# Corpus

The public façade. One class, ~40 methods, the only thing most Python users need to import.

```python
from kglite_docs import Corpus

with Corpus.open("kb.kgl") as c:
    hits = c.search("query")
```

::: kglite_docs.corpus.Corpus
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members_order: source
      heading_level: 2
