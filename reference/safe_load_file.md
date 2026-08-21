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
#> -0.8637 -0.2746 -0.3624  2.0918 -1.0524 -0.5973 -0.2616  1.6429  0.2105 -0.1486
#> -0.5188 -1.6553 -1.0440 -1.6912  0.2292 -1.3591 -0.3557 -0.6572  0.8490 -1.3458
#> -0.4858  0.1247  0.1203  0.4919  0.0160  0.2224  0.4879 -0.8437  0.3096 -0.0019
#>  0.6604 -0.2942 -0.3267 -0.0900  1.3291  0.6672  0.0493 -1.6434  1.0731  1.3468
#> -0.0917 -0.2034  0.5318 -0.0577 -2.0781 -0.9139  0.6811  0.6756  0.2803 -0.1451
#>  0.3310  0.1290 -1.0286  0.0082 -0.1891 -0.7608  1.4076 -0.3867  0.7913 -2.3949
#> -0.4580  0.0263 -1.9580  0.2076 -0.6674 -0.3674  1.0195 -0.5340  0.2517  0.6449
#> -0.8908 -0.7580 -1.9289  0.3894  0.3800  2.3520  0.8000  0.2757  0.3724  0.8248
#>  0.6557  0.8680 -0.2699  1.1264  0.1444 -0.1440 -0.3759 -0.0329  0.7592  2.1010
#> -0.6555  0.6583  0.3219  0.9465  1.1885 -0.7816  0.7993  0.6244 -1.0973 -0.5275
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
