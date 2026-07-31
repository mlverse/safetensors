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
#> -1.0522 -0.3105 -0.7004 -0.2855 -0.7978 -1.5543  0.6825  1.2383 -2.2750 -0.1284
#> -0.2066  0.2990 -0.1297 -0.0088  0.7688  1.7114 -0.8003  0.0556  0.1949  1.0138
#> -2.0188 -0.5986 -0.4113  0.3381  0.6541 -0.9203 -1.0475  1.0630  0.7825 -0.1169
#> -1.3578  1.5953  0.2034 -1.7798  0.3606  0.5834 -0.5821 -0.6654  1.2621 -1.0519
#>  0.2910  0.2068  0.4514 -0.0109  0.1481  0.0751  0.0151 -0.6037  1.0949  1.4688
#>  1.2814  0.1667  0.1124 -0.1155 -0.4915 -1.2927  0.4691 -1.8635  0.6899 -0.3076
#> -0.3274  0.2020 -0.5428  1.0272  0.1713  0.9865  0.7827  1.5063 -0.6522 -1.9105
#> -0.2332  0.9754  0.8582  1.8241  2.3164  0.4673  0.7453  2.0340  0.3546 -0.9347
#>  0.3735 -0.0363  0.6432 -2.0167 -1.2825  1.8455  0.1105  0.0246  0.1680 -1.1913
#>  0.7768 -0.4875  0.7416 -0.1273 -1.3631  0.2209  0.5459 -0.9558  0.0208  0.4717
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
