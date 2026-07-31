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
#>  0.3165  1.9106  1.6233  1.0397  0.7997  1.4554  1.2137 -0.3235  0.7583  1.4878
#> -0.3745  0.6842  0.5345  1.0122 -1.0464 -1.0745  0.7580 -1.5171  0.8294  0.9891
#>  0.8480  0.6991 -1.6063 -0.8475 -0.6525  0.6142  1.4050  1.3013  0.2089 -0.5648
#>  0.1586 -0.4146  0.5516 -0.6425  1.7115  1.0994  0.4958 -0.1314 -1.6834  0.6578
#>  0.5240  0.1921 -0.4970 -0.5278 -1.6023  1.4728 -0.0617  1.5684  0.5315 -0.7505
#> -0.7397 -1.7135 -1.0177  0.2802  1.2844 -0.6086 -1.0286  1.5318  1.1030 -0.2320
#>  0.9957  0.1297 -0.0291  1.3193  1.0100  0.4470 -0.0856 -0.5905 -0.2495 -0.7879
#> -1.7483 -0.2623  0.5436 -0.9183 -0.1074 -1.2908 -0.3358 -0.4143  0.0335 -0.3657
#>  1.4957  0.5394  1.4908  0.6511 -0.4592  0.0781 -0.2710 -0.2142  0.1106  0.7529
#>  1.5983 -0.3535  0.4388 -1.0293  0.8711 -1.0376 -0.6748  0.4201  1.8837  0.7944
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
