# Writes a list of tensors to the safetensors format

Writes a list of tensors to the safetensors format

## Usage

``` r
safe_save_file(tensors, path, ..., metadata = NULL)

safe_serialize(tensors, ..., metadata = NULL)
```

## Arguments

- tensors:

  A named list of tensors. Currently only torch tensors are supported.

- path:

  The path to save the tensors to. It can also be a binary connection,
  as eg, created with
  [`file()`](https://rdrr.io/r/base/connections.html).

- ...:

  Currently unused.

- metadata:

  An optional string that is added to the file header. Possibly adding
  additional description to the weights.

## Value

The path invisibly or a raw vector.

## Functions

- `safe_serialize()`: Serializes the tensors and returns a raw vector.

## Examples

``` r
if (rlang::is_installed("torch") && torch::torch_is_installed()) {
  tensors <- list(x = torch::torch_randn(10, 10))
  temp <- tempfile()
  safe_save_file(tensors, temp)
  safe_load_file(temp, framework = "torch")

  ser <- safe_serialize(tensors)
}
```
