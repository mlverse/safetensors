# safetensors 0.2.0

* Refactored the package so other packages can extend it.
  This is possible by:
  1. Implementing the `safe_tensor_buffer` and `safe_tensor_meta` methods.
  2. Registering the framework with the reflection `safetensors_frameworks`.

* `safe_load_file` no longer defines a default framework. Set `framework='torch'` to get the previous behavior.
# safetensors 0.1.2

* Added support for BF16 data types.

# safetensors 0.1.1

* Added a `NEWS.md` file to track changes to the package.
* Copy tensors to cpu before proceeding with serialization. (#2)
