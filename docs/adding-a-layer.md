# Adding a layer

Every layer is a small Python class in [`backend/brickdnn/layers.py`](../backend/brickdnn/layers.py).
The UI builds its palette and forms from `GET /api/layers/`, so **no frontend changes are
needed**.

As an example, here is `GroupNorm`:

```python
@register
class GroupNorm(Layer):
    type = "GroupNorm"                  # identifier used in graph JSON
    label = "GroupNorm"                 # name in the palette
    category = "Normalization"          # one of CATEGORIES
    description = "Normalises groups of channels; works well with small batches."
    params = (
        Param("num_groups", "Groups", "int", default=8, minimum=1),
        Param("num_channels", "Channels", "int", auto=True, minimum=1,
              help="Leave empty to infer it from the input shape."),
    )
    name_prefix = "groupnorm"           # attribute names: groupnorm_1, groupnorm_2, ...

    def infer(self, inputs):
        # Validate the input and return the output shape (batch dimension excluded).
        shape = self._single(inputs, (2, 3))          # (C, L) or (C, H, W)
        channels = self._auto("num_channels", shape[0], "channels")
        if channels % self.p["num_groups"]:
            raise LayerError(
                f"{channels} channels cannot be split into {self.p['num_groups']} groups."
            )
        return shape

    def num_params(self):
        return 2 * self.p["num_channels"]

    def init_code(self):
        return f"nn.GroupNorm({self.p['num_groups']}, {self.p['num_channels']})"
```

## The building blocks

- **`Param`**: `kind` is `int`, `float`, `bool`, `select` (with `options`) or `ints`
  (with an optional fixed `length`). Values are validated and converted before `infer` runs,
  and `self.p` holds them.
- **`auto=True`** parameters may be empty. Call `self._auto(name, inferred, description)` in
  `infer` to fill them in. If the user typed a different value, it raises a helpful error.
- **`self._single(inputs, ranks)`** returns the only input shape and checks its rank.
- **`infer`** must raise `LayerError` with a message written for users when something does not
  fit. The message is shown on the canvas.
- **`num_params`** must match `sum(p.numel() for p in module.parameters())` exactly; the tests
  check this against PyTorch.
- **`init_code`** returns the constructor expression (it may span several lines).
- Override **`forward_lines(attr, inputs, out)`** if the module is not called as
  `out = self.attr(x)` (see `_Recurrent` and `PositionalEmbedding`). Layers without weights can
  set `has_module = False` and only produce forward code (see `Add`, `Reshape`).
- Multi-input layers set `min_inputs`/`max_inputs` (`None` = unlimited).
- Layers that take integer token ids set `input_dtypes = ("long",)`.

## Tests

Add at least one entry to `CASES` in
[`backend/tests/test_layers.py`](../backend/tests/test_layers.py):

```python
((16, 8, 8), "float", "GroupNorm", {"num_groups": 4}, (16, 8, 8)),
```

This one line checks the shape inference **and**, through `test_codegen.py`, generates the
code, runs it in PyTorch and compares output shape and parameter count. Add error cases to
`test_invalid_inputs` as well. Then run:

```bash
cd backend
pip install torch --index-url https://download.pytorch.org/whl/cpu   # once
pytest
```

Optionally, give the layer a nicer one-line summary on the canvas in
[`frontend/src/builder/format.ts`](../frontend/src/builder/format.ts) (`summarize`).
