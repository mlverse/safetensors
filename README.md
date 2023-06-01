
<!-- README.md is generated from README.Rmd. Please edit that file -->

# safetensors

<!-- badges: start -->

[![R-CMD-check](https://github.com/mlverse/safetensors/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mlverse/safetensors/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

safetensors is a pure R implementation of the
[safetensors](https://github.com/huggingface/safetensors) file format.

Currently only reading files is supported.

## Installation

You can install the development version of safetensors from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("mlverse/safetensors")
```

## Example

Here’s an example of writing and reading safetensors files:

``` r
library(torch)
library(safetensors)

tensors <- list(
  x = torch_randn(10, 10),
  y = torch_ones(10, 10)
)

str(tensors)
#> List of 2
#>  $ x:Float [1:10, 1:10]
#>  $ y:Float [1:10, 1:10]

tmp <- tempfile()
safe_save_file(tensors, tmp)

tensors <- safe_load_file(tmp)
str(tensors)
#> List of 2
#>  $ x:Float [1:10, 1:10]
#>  $ y:Float [1:10, 1:10]
```
