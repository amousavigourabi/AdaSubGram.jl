module Hashing

"""
    fnv_1a_hash(word::String, n::UInt32) -> UInt32

Hashes the word to a 32-bit space,
using the FNV 1A hashing algorithm.
Shifts the output by one to ensure
we can always index arrays. This
means that n=2_000_000 will ensure
the result is included in the range
[1, 2_000_000], instead of the
range [0, 1_999_999].
"""
function fnv_1a_hash(word::String, n::UInt32)::UInt32
  offset = UInt32(2166136261)
  prime = UInt32(16777619)
  hash = offset
  for byte in codeunits(word)
    hash = hash ⊻ UInt32(byte)
    hash = hash * prime
  end
  return hash % n + 1
end

"""
    hash_words(words::Vector{String}, n::UInt32) -> Vector{UInt32}

Hashes all words to a 32-bit space,
using the FNV 1A hashing algorithm.
"""
function hash_words(words::Vector{String}, n::UInt32)::Vector{UInt32}
  return fnv_1a_hash.(words, n)
end

export hash_words

end
