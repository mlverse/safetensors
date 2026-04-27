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
#>  0.2125 -1.0636  0.9769 -2.3091 -1.5507 -0.4881 -1.3005  0.0343 -0.9570  0.0840
#>  0.0882 -0.9028 -1.7188 -0.5232  0.3155  1.3846  0.4263 -0.2140  1.9444 -0.3422
#>  0.3376 -0.6911 -0.2339  0.3735  1.0596 -0.8985  2.2322  1.1563 -0.7763 -2.2505
#> -0.5391 -0.8767  0.8312 -0.6931  0.0264 -0.5513 -0.6931 -1.9926  1.6548  0.3357
#>  0.1842  1.2137 -1.2100  0.3595  0.1223 -0.4476  0.6559 -2.9581 -0.9653 -0.2293
#> -0.3156  0.1600  0.6620  1.8815  0.6285  0.0352  0.6753 -0.4303 -0.2496 -0.0590
#>  0.3410 -1.1894  0.8972  0.5174 -0.4926  0.2055 -0.1823  0.2798 -1.0694 -1.3928
#> -1.3731 -1.2850 -0.4928  1.0496  1.4339  0.2832 -1.1369 -0.3949 -0.0604  1.8323
#> -0.0105  1.0371  1.0700 -0.6941  1.3153  0.2477 -0.2038 -1.7772  1.3704  1.4911
#>  1.1020 -0.4392  0.3307 -2.2715 -1.0826  0.9690  1.5879 -0.3231 -1.5693  0.7261
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
