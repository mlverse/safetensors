test_that("load a file", {
  dict <- safe_load_file(test_path("safetensors/hello.safetensors"))
  expect_equal(names(dict), c("hello", "world"))

  expect_equal(dict$hello$shape, c(10, 10))
  expect_true(all(as.numeric(dict$hello) == 1))

  expect_equal(dict$world$shape, c(5, 10))
  expect_true(all(as.numeric(dict$world) == 0))
})
