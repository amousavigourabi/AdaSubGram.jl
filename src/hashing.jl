module Hashing

function fnv_1a_hash(word::String, n::UInt32)::UInt32
  offset = UInt32(2166136261)
  prime = UInt32(16777619)
  hash = offset
  for byte in codeunits(word)
    hash = hash ⊻ UInt32(byte)
    hash = hash * prime
  end
  return hash % n
end

function hash_words(words::Vector{String}, n::UInt32)::Vector{UInt32}
  return fnv_1a_hash.(words, n)
end

export hash_words

end
