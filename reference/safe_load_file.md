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
#>  2.3907  2.1660  0.2028 -0.6311  0.9642 -1.0962 -1.3181  1.2580 -0.7042  0.2751
#>  0.3312 -0.2691 -0.1724  1.3593 -1.7377  0.0273 -0.4729 -0.8000  1.0007 -0.0699
#> -1.6437 -0.2418 -0.2062 -0.7982  0.7364  0.8544 -1.2423 -0.8648  0.7463 -0.3324
#>  0.9774  1.3175  0.6996  0.9685  0.4946  0.7957  1.0366 -1.1282  0.7936 -0.7011
#> -1.0798  0.2614  0.5060 -1.2590 -0.0206  0.9597  0.0412  1.1162 -0.2332  0.2802
#>  0.5989  2.3277 -1.2594  0.4232 -1.0517  0.7951 -0.0407  0.3080  0.0452 -0.1954
#>  0.9022 -0.0723 -1.5617  0.8894  0.1179 -1.1895 -1.5009 -1.4739 -0.9761  0.3972
#> -0.6898 -0.4799  0.2467 -1.3958  0.6558 -1.2297 -1.0873 -0.5249  1.8593  0.7921
#> -1.6207 -1.4700 -1.5692  0.8707 -0.9051  0.2889  1.0315  0.0482  1.1883  0.2493
#> -1.1267 -0.5119  0.8245  0.6758  1.5415 -0.5460 -0.5762 -0.9591 -1.0686 -0.4259
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
