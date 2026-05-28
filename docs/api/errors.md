# Errors

Catch [`KgliteDocsError`][kglite_docs.errors.KgliteDocsError] for anything raised by the library. The concrete subclasses let you distinguish *why* a call failed.

```python
from kglite_docs import KgliteDocsError, SelfVerificationError

try:
    corpus.verify_summary(sid, verdict="verified", verifier_agent_id="me")
except SelfVerificationError:
    # I authored this summary — get a different agent to verify
    ...
except KgliteDocsError as exc:
    # any other library-level failure
    ...
```

::: kglite_docs.errors
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members_order: source
