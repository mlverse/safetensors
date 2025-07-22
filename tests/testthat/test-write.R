test_that("can write a safetensors file (torch)", {
  skip_if_not_installed("torch")

  tensors <- list(
    x = torch::torch_randn(10, 10),
    y = torch::torch_randn(5, 5)
  )

  tmp <- tempfile(fileext = ".safetensors")
  safe_save_file(tensors, tmp)

  reloaded <- safe_load_file(tmp, framework = "torch")

  expect_true(torch::torch_allclose(tensors$x, reloaded$x))
  expect_true(torch::torch_allclose(tensors$y, reloaded$y))
})

test_that("can write a safetensors file (pjrt)", {
  skip_if_not_installed("pjrt")
  skip_on_os("windows")

  buffers <- list(
    x = pjrt::pjrt_buffer(array(rnorm(100), dim = c(10, 10))),
    y = pjrt::pjrt_buffer(array(1:20, dim = c(4, 5)))
  )

  tmp <- tempfile(fileext = ".safetensors")
  safe_save_file(buffers, tmp)

  reloaded <- safe_load_file(tmp, framework = "pjrt")

  expect_true(identical(pjrt::as_array(buffers$x), pjrt::as_array(reloaded$x)))
  expect_true(identical(pjrt::as_array(buffers$y), pjrt::as_array(reloaded$y)))
})

test_that("with different datatypes (torch)", {
  data_type <- c(
    "float16",
    "float",
    "float64",
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64"
  )

  for (dtype in data_type) {
    x <- list(x = torch::torch_randn(10)$to(dtype = dtype))

    tmp <- tempfile(fileext = ".safetensors")
    safe_save_file(x, tmp)

    reloaded <- safe_load_file(tmp, framework = "torch")

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

test_that("with different datatypes (pjrt)", {
  skip_if_not_installed("pjrt")
  skip_on_os("windows")
  types <- list(
    list(pjrt_type = "f32", rtype = "double"),
    list(pjrt_type = "f64", rtype = "double"),
    list(pjrt_type = "s8", rtype = "integer"),
    list(pjrt_type = "s16", rtype = "integer"),
    list(pjrt_type = "s32", rtype = "integer"),
    list(pjrt_type = "s64", rtype = "integer"),
    list(pjrt_type = "u8", rtype = "integer"),
    list(pjrt_type = "u16", rtype = "integer"),
    list(pjrt_type = "u32", rtype = "integer"),
    list(pjrt_type = "u64", rtype = "integer"),
    list(pjrt_type = "pred", rtype = "logical")
  )

  dat <- c(0L, 1:8, 0L)
  for (type in types) {
    x <- switch(
      type$rtype,
      double = as.double(dat),
      integer = as.integer(dat),
      logical = as.logical(dat),
      stop()
    )

    x <- list(
      x = pjrt::pjrt_buffer(array(x, dim = c(5, 2)), type = type$pjrt_type)
    )

    tmp <- tempfile(fileext = ".safetensors")
    safe_save_file(x, tmp)

    reloaded <- safe_load_file(tmp, framework = "pjrt")

    expect_true(identical(pjrt::as_array(x$x), pjrt::as_array(reloaded$x)))
  }
})
