module Preprocessing

using Unicode

function normalize(word::String)
  return Unicode.normalize(word, :NFC)
end

function split_subwords(word::String, s_min::Int64, s_max::Int64)
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

export normalize, split_subwords

end
