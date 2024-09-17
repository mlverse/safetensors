test_that("R-level cli_abort messages are correctly translated in FR", {

  withr::with_language(lang = "fr",
                       expect_error(
                         safe_save_file(list(x = torch::torch_randn(10, 10), x = torch::torch_randn(10, 10))),
                         regexp = "Les noms dupliqués ne sont pas autorisés",
                         fixed = TRUE
                       ))
  })
