test_that("load a file (torch)", {
  skip_if_not_installed("torch")
  dict <- safe_load_file(test_path("safetensors/hello.safetensors"), framework = "torch")
  expect_equal(names(dict), c("hello", "world"))

  expect_equal(dict$hello$shape, c(10, 10))
  expect_true(all(as.numeric(dict$hello) == 1))

  expect_equal(dict$world$shape, c(5, 10))
  expect_true(all(as.numeric(dict$world) == 0))
})

test_that("load a file (pjrt)", {
  skip_if_not_installed("pjrt")
  dict <- safe_load_file(test_path("safetensors/hello.safetensors"), framework = "pjrt")
  expect_equal(names(dict), c("hello", "world"))

  expect_equal(dim(dict$hello), c(10, 10))
  expect_true(all(pjrt::as_array(dict$hello) == 1))

  expect_equal(dim(dict$world), c(5, 10))
  expect_true(all(pjrt::as_array(dict$world) == 0))
})
