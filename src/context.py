import contextvars

context_sample = contextvars.ContextVar("context_sample", default=None)
