module Preprocessing

using Unicode

"""
    split_subwords(word::String, s_min::Int64, s_max::Int64) -> Vector{String}

Return a Vector of all subwords of sizes s_min to s_max (inclusive)
that can be found in the provided word, with added boundary characters.

s_max must be greater than or equal to s_min.
"""
function split_subwords(word::String, s_min::Int64, s_max::Int64)::Vector{String}
  enclosed_word = collect(Unicode.graphemes("<" * word * ">"))
  n_grams = []
  for s_i in s_min:s_max
    for start in 1:(length(enclosed_word)-s_i+1)
      n_gram = join(enclosed_word[start:(start+s_i-1)])
      push!(n_grams, n_gram)
    end
  end
  return n_grams
end

"""
    normalize_encodings(document::String) -> String

Normalizes the Unicode string input to an NFC-normalized equivalent.
"""
function normalize_encodings(document::String)::String
  return Unicode.normalize(document, :NFC)
end

"""
    remove_punctuation(document::String) -> String

Removes all punctuation from the input string.
"""
function remove_punctuation(document::String)::String
  return filter(!ispunct, document)
end

"""
    to_lowercase(document::String) -> String

Transforms all characters in the input string to lowercase.
"""
function to_lowercase(document::String)::String
  return lowercase(document)
end

"""
    normalize(document::String) -> String

Normalizes the input string by removing all punctuation,
converting the character encodings to an equivalent
NFC form, and making all characters lowercase.
"""
function normalize(document::String)::String
  return to_lowercase(normalize_encodings(remove_punctuation(document)))
end

"""
    tokenize(document::String) -> Vector{String}

Returns the tokens from a document. This is
done by splitting the document on whitespace.
"""
function tokenize(document::String)::Vector{String}
  return split(document)
end

export normalize, split_subwords

end
