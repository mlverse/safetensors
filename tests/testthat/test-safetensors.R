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

test_that("roundtrip file with empty string tensor name (torch)", {
  skip_if_not_installed("torch")

  # Empty string keys commonly occur when using PyTorch forward hooks to capture
  # intermediate outputs. For example, when validating a native R torch UNet
  # implementation against Python diffusers, the root module output is captured
  # with name "".
  temp <- tempfile(fileext = ".safetensors")
  tensors <- list(torch::torch_ones(2, 3), named = torch::torch_zeros(3, 2))
  names(tensors)[1] <- ""  # Empty string key (root module output)

  # Write should succeed
  expect_no_error(safe_save_file(tensors, temp))

  # Read should succeed
  dict <- safe_load_file(temp, framework = "torch")

  # Verify empty key is preserved
  expect_true("" %in% names(dict))

  # Note: R's list[[""]] returns NULL, so access empty keys by position
  empty_idx <- which(names(dict) == "")
  expect_equal(length(empty_idx), 1)

  # Verify tensor contents
  empty_tensor <- dict[[empty_idx]]
  expect_s3_class(empty_tensor, "torch_tensor")
  expect_equal(empty_tensor$shape, c(2, 3))
  expect_true(all(as.numeric(empty_tensor) == 1))

  # Named tensors should work normally
  expect_equal(dict$named$shape, c(3, 2))
  expect_true(all(as.numeric(dict$named) == 0))
})

test_that("empty tensor name coexists with metadata and escaped keys (torch)", {
  skip_if_not_installed("torch")

  # An empty key must not break the __metadata__ path, and the hand-built
  # JSON used when an empty key is present must escape other keys.
  temp <- tempfile(fileext = ".safetensors")
  tensors <- list(torch::torch_ones(2), torch::torch_zeros(2))
  names(tensors) <- c("", "a\"b\\c")

  expect_no_error(safe_save_file(tensors, temp, metadata = list(fmt = "pt")))
  dict <- safe_load_file(temp, framework = "torch")

  expect_true(all(c("", "a\"b\\c") %in% names(dict)))
  expect_equal(attr(dict, "metadata")[["__metadata__"]]$fmt, "pt")

  # metadata path with only ordinary keys (regression: the name restore
  # must not fire while the __metadata__ slot makes meta_ longer than nms)
  temp2 <- tempfile(fileext = ".safetensors")
  expect_no_error(safe_save_file(
    list(a = torch::torch_ones(2)), temp2, metadata = list(fmt = "pt")
  ))
  d2 <- safe_load_file(temp2, framework = "torch")
  expect_equal(attr(d2, "metadata")[["__metadata__"]]$fmt, "pt")
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
