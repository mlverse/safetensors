test_that("load a file (torch)", {
  skip_if_not_installed("torch")
  dict <- safe_load_file(
    test_path("safetensors/hello.safetensors"),
    framework = "torch"
  )
  expect_equal(names(dict), c("hello", "world"))

  expect_equal(dict$hello$shape, c(10, 10))
  expect_true(all(as.numeric(dict$hello) == 1))

  expect_equal(dict$world$shape, c(5, 10))
  expect_true(all(as.numeric(dict$world) == 0))
})

#test_that("torch & pjrt interoperability", {
#  skip_if_not_installed("torch")
#  skip_if_not_installed("pjrt")
#  skip_on_os("windows")
#
#  x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
#  torch_tensors <- list(x = torch::torch_tensor(x))
#  pjrt_buffers <- list(x = pjrt::pjrt_buffer(x))
#
#  temp1 <- tempfile()
#  temp2 <- tempfile()
#
#  safe_save_file(torch_tensors, temp1)
#  safe_save_file(pjrt_buffers, temp2)
#
#  x1 <- safe_load_file(temp1, framework = "torch")
#  x2 <- safe_load_file(temp2, framework = "pjrt")
#  x3 <- safe_load_file(temp1, framework = "pjrt")
#  x4 <- safe_load_file(temp2, framework = "torch")
#
#  expect_equal(torch::as_array(x1$x), pjrt::as_array(x2$x))
#  expect_equal(pjrt::as_array(x2$x), pjrt::as_array(x3$x))
#  expect_equal(pjrt::as_array(x3$x), torch::as_array(x4$x))
#})
