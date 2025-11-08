module Preprocessing

using Unicode

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
  return normalize_encodings(document)
end

"""
    tokenize(document::String) -> Vector{String}

Returns the tokens from a document. This is
done by splitting the document on whitespace.
"""
function tokenize(document::String)::Vector{String}
  return split(document)
end

export normalize, tokenize, to_lowercase, remove_punctuation

end
