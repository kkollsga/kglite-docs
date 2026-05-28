# Types

`Literal` aliases for the enum-ish string args, and `TypedDict`s for
dict-shaped return values. Importing from here gives you IDE
autocomplete on result fields and rejects typos in arguments.

```python
from kglite_docs.types import Verdict, SearchHit

def my_fn(verdict: Verdict) -> list[SearchHit]:
    ...
```

::: kglite_docs.types
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members_order: source
