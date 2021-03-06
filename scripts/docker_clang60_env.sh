# Use clang instead of gcc
export OMPI_CC=clang
export OMPI_CXX=clang++
export CC=mpicc
export CXX=mpicxx

# Add clang OpenMP runtime library to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

SANITIZER=$1
if [ "$SANITIZER" == "undefined_sanitizer" ]
  then FLAGS="-fsanitize=address -fsanitize=undefined -fsanitize-blacklist=/scratch/source/trilinos/release/DataTransferKit/scripts/undefined_blacklist.txt"
  # Suppress leak sanitizer on functions we have no control of.
  export LSAN_OPTIONS=suppressions=/scratch/source/trilinos/release/DataTransferKit/scripts/leak_blacklist.txt
  # Set the ASAN_SYMBOLIZER_PATH environment variable to point to the
  # llvm-symbolizer to make AddressSanitizer symbolize its output.  This makes
  # the reports more human-readable.
  export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-5.0/bin/llvm-symbolizer
elif [ "$SANITIZER" == "thread_sanitizer" ]
  then FLAGS="-fsanitize=thread"
fi

# Use SANITIZER_FLAGS to add Wextra flags to avoid increasing the complexity of
# our script. We don't put Wextra directly in docker_cmake because it would
# affect every build and the number of warnings is very large.
export SANITIZER_FLAGS="-Wextra $FLAGS"
