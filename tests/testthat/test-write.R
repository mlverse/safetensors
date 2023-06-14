test_that("can write a safetensors file", {

  tensors <- list(
    x = torch::torch_randn(10, 10),
    y = torch::torch_randn(5, 5)
  )

  tmp <- tempfile(fileext = ".safetensors")
  safe_save_file(tensors, tmp)

  reloaded <- safe_load_file(tmp)

  expect_true(torch::torch_allclose(tensors$x, reloaded$x))
  expect_true(torch::torch_allclose(tensors$y, reloaded$y))
})

test_that("with different datatypes", {

  data_type <- c("float16",
                 "float",
                 "float64",
                 "bool",
                 "uint8",
                 "int8",
                 "int16",
                 "int32",
                 "int64")

  for (dtype in data_type) {
    x <- list(x = torch::torch_randn(10)$to(dtype=dtype))

    tmp <- tempfile(fileext = ".safetensors")
    safe_save_file(x, tmp)

    reloaded <- safe_load_file(tmp)

    expect_true(torch::torch_allclose(x$x, reloaded$x))
  }

})

test_that("metadata validations", {

  tensors <- list(
    x = torch::torch_randn(10, 10),
    y = torch::torch_randn(5, 5)
  )

  tmp <- tempfile(fileext = ".safetensors")

  metadata <- 1
  expect_snapshot_error({
    safe_save_file(tensors, tmp, metadata = metadata)
  })

  metadata <- list(x = 1)
  expect_snapshot_error({
    safe_save_file(tensors, tmp, metadata = metadata)
  })

  metadata <- list(y = c("1", "2"))
  expect_snapshot_error({
    safe_save_file(tensors, tmp, metadata = metadata)
  })

  metadata <- list("a")
  expect_snapshot_error({
    safe_save_file(tensors, tmp, metadata = metadata)
  })

})
