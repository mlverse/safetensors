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
