# Changelog

## safetensors (development version)

- Added support for reading and writing `F8_E4M3` and `F8_E5M2` tensors
  with torch. ([\#13](https://github.com/mlverse/safetensors/issues/13),
  1)

- Fixed reading and writing tensors with empty names. Because `x[[""]]`
  returns `NULL` for a named R list, access these tensors by position.
  ([\#10](https://github.com/mlverse/safetensors/issues/10),

  1.  

- Fixed reading and writing tensors at offsets larger than 2 GB.
  ([\#14](https://github.com/mlverse/safetensors/issues/14),

  1.  

- Fixed writing `bfloat16` tensors.
  ([\#11](https://github.com/mlverse/safetensors/issues/11), 1)

## safetensors 0.2.1

CRAN release: 2026-04-27

- Changed maintainer to Tomasz Kalinowski.

## safetensors 0.2.0

CRAN release: 2025-08-18

- Refactored the package so other packages can extend it. This is
  possible by:

  1.  Implementing the `safe_tensor_buffer` and `safe_tensor_meta`
      methods.
  2.  Registering the framework with the reflection
      `safetensors_frameworks`.

- `safe_load_file` no longer defines a default framework. Set
  `framework='torch'` to get the previous behavior. \# safetensors 0.1.2

- Added support for BF16 data types.

## safetensors 0.1.1

CRAN release: 2023-07-10

- Added a `NEWS.md` file to track changes to the package.
- Copy tensors to cpu before proceeding with serialization.
  ([\#2](https://github.com/mlverse/safetensors/issues/2))
