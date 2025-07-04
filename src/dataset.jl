module Dataset

using Unicode

"""
    assign_labels!(document::Vector{String}, labels::Dict{String, UInt64}, words::Vector{String}, counts::Vector{UInt64}, current_label::UInt64) -> Tuple{Vector{UInt64}, UInt64}

Converts the document into a vector of integer labels.
Adds any potential new labels to the labels dictionary.
Keeps track of and returns the current label and
increments it when necessary. The vector words
maintains the reverse mapping from indices to words.
"""
function assign_labels!(document::Vector{String}, labels::Dict{String, UInt64}, words::Vector{String}, counts::Vector{UInt64}, current_label::UInt64)::Tuple{Vector{UInt64}, UInt64}
  labelled_document = Vector{UInt64}(undef, length(document))
  for (i, token) in enumerate(document)
    if haskey(labels, token)
      labelled_document[i] = labels[token]
      counts[labels[token]] += 1
    else
      labelled_document[i] = current_label
      labels[token] = current_label
      words[current_label] = token
      counts[current_label] += 1
      current_label += 1
    end
  end
  return labelled_document, current_label
end

"""
    pair_contexts(document::Vector{UInt64}, c::UInt8) -> Vector{Tuple{UInt64, Vector{UInt64}}}

Creates token to context window pairings from
the list of integer labels. The context size c
is used symmetrically, to index-c and index+c.
"""
function pair_contexts(document::Vector{UInt64}, c::UInt8)::Vector{Tuple{UInt64, Vector{UInt64}}}
  contexts = Vector{Tuple{UInt64, Vector{UInt64}}}(undef, length(document))
  for (i, token) in enumerate(document)
    start = max(i - c, 1)
    final = min(i + c, length(document))
    context = document[[start:(i-1); (i+1):final]]
    contexts[i] = (token, context)
  end
  return contexts
end

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

export assign_labels!, pair_contexts, split_subwords

end
