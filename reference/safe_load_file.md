# Safe load a safetensors file

Loads an safetensors file from disk.

## Usage

``` r
safe_load_file(path, ..., framework)
```

## Arguments

- path:

  Path to the file to load

- ...:

  Additional framework dependent arguments to pass to the tensor
  creation function.

- framework:

  Framework to load the data into. Currently supports "torch" and "pjrt"

## Value

A list with tensors in the file. The `metadata` attribute can be used to
find metadata the metadata header in the file.

## See also

[safetensors](https://mlverse.github.io/safetensors/reference/safetensors.md),
[`safe_save_file()`](https://mlverse.github.io/safetensors/reference/safe_save_file.md)

## Examples

``` r
if (rlang::is_installed("torch") && torch::torch_is_installed()) {
  tensors <- list(x = torch::torch_randn(10, 10))
  temp <- tempfile()
  safe_save_file(tensors, temp)
  safe_load_file(temp, framework = "torch")
}
#> $x
#> torch_tensor
#>  0.2143  0.0066  1.6714 -0.6687  0.1124  0.4103  0.1947 -1.6104  0.2122  0.7684
#> -0.6392 -0.0518 -1.6110 -0.6797  0.2716 -0.2045  0.7152 -0.4711 -1.1840  0.9666
#>  0.1254 -0.0439 -1.0692  0.2276  0.2484  1.6495 -1.0974  0.7842  0.5288  0.8137
#> -0.2152 -0.6730  0.9458 -1.0982 -1.2524  1.9018 -1.3751  0.2073  0.7661  0.4846
#> -0.2175 -1.1032  0.8200 -0.9400 -0.4754 -1.2146  0.3053 -0.1859  0.5275  0.6718
#>  0.4593 -0.5022  0.0640  0.8759 -2.4450  0.7306 -1.5706  0.5020 -0.4612  0.7317
#>  0.6408 -0.3295 -0.2860 -0.3918 -0.1788 -0.7001 -1.3475  0.9457  0.2345 -0.1842
#> -1.5587 -0.4584  0.3074 -0.3054  0.8307  0.7954  0.6528  0.8436 -0.0037  0.4590
#>  0.0226 -0.1655  0.1507 -3.3422  0.6700 -0.8959 -2.0288 -0.9959 -0.4704 -1.6428
#> -0.3453 -0.1433 -0.1779  1.2169  1.9186 -0.5070 -0.7354  0.2406 -0.1660 -0.4255
#> [ CPUFloatType{10,10} ]
#> 
#> attr(,"metadata")
#> attr(,"metadata")$x
#> attr(,"metadata")$x$shape
#> [1] 10 10
#> 
#> attr(,"metadata")$x$dtype
#> [1] "F32"
#> 
#> attr(,"metadata")$x$data_offsets
#> [1]   0 400
#> 
#> 
#> attr(,"max_offset")
#> [1] 468
```
