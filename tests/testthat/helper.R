create_test_tensors <- function(framework, dtype = NULL, shape = c(10, 10)) {
  if (framework == "torch") {
    if (is.null(dtype)) {
      tensors <- list(
        x = torch::torch_randn(shape),
        y = torch::torch_randn(rev(shape))
      )
    } else {
      tensors <- list(
        x = torch::torch_randn(shape)$to(dtype = dtype)
      )
    }
  } else if (framework == "pjrt") {
    if (is.null(dtype)) {
      # Default to f32 for pjrt
      dtype <- "f32"
    }

    # Create appropriate data based on dtype
    if (dtype == "pred") {
      data_x <- array(sample(c(TRUE, FALSE), prod(shape), replace = TRUE), dim = shape)
      data_y <- array(sample(c(TRUE, FALSE), prod(rev(shape)), replace = TRUE), dim = rev(shape))
    } else if (dtype %in% c("s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64")) {
      data_x <- array(sample(1:100, prod(shape), replace = TRUE), dim = shape)
      data_y <- array(sample(1:100, prod(rev(shape)), replace = TRUE), dim = rev(shape))
    } else if (dtype %in% c("f32", "f64")) {
      data_x <- array(rnorm(prod(shape)), dim = shape)
      data_y <- array(rnorm(prod(rev(shape))), dim = rev(shape))
    } else {
      stop()
    }

    tensors <- list(
      x = pjrt::pjrt_buffer(data_x, type = dtype),
      y = pjrt::pjrt_buffer(data_y, type = dtype)
    )
  }

  tensors
}

#' Compare tensors for equality
#'
#' @param tensor1 First tensor
#' @param tensor2 Second tensor
#' @param framework Framework used ("torch" or "pjrt")
#' @param tolerance Tolerance for floating point comparison
#' @return TRUE if tensors are equal
compare_tensors <- function(tensor1, tensor2, framework, tolerance = 1e-6) {
  if (framework == "torch") {
    torch::torch_allclose(tensor1, tensor2, atol = tolerance)
  } else if (framework == "pjrt") {
    # Convert to arrays and compare
    arr1 <- pjrt::as_array(tensor1)
    arr2 <- pjrt::as_array(tensor2)
    all.equal(arr1, arr2, tolerance = tolerance)
  }
}

#' Get supported datatypes for a framework
#'
#' @param framework Either "torch" or "pjrt"
#' @return Character vector of supported datatypes
get_supported_dtypes <- function(framework) {
  if (framework == "torch") {
    c("float16", "float", "float64", "bool", "uint8", "int8", "int16", "int32", "int64", "bfloat16")
  } else if (framework == "pjrt") {
    c("pred", "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64",
      "f16", "f32", "f64", "bf16", "c64", "c128")
  }
}
