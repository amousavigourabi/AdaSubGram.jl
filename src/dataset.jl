module Dataset

using Random
using Unicode
using AdaSubGram

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
      @inbounds labelled_document[i] = labels[token]
      @inbounds counts[labels[token]] += 1
    else
      @inbounds labelled_document[i] = current_label
      labels[token] = current_label
      push!(words, token)
      push!(counts, 1)
      current_label += 1
    end
  end
  return labelled_document, current_label
end

"""
    pair_contexts(document::Vector{UInt64}, c::Int64) -> Vector{Tuple{UInt64, Vector{UInt64}}}

Creates token to context window pairings from
the list of integer labels. The context size c
is used symmetrically, to index-c and index+c.
"""
function pair_contexts(document::Vector{UInt64}, c::Int64)::Vector{Tuple{UInt64, Vector{UInt64}}}
  contexts = Vector{Tuple{UInt64, Vector{UInt64}}}(undef, length(document))
  for (i, token) in enumerate(document)
    @inbounds context = document[[max(i - c, 1):(i-1); (i+1):min(i + c, end)]]
    @inbounds contexts[i] = (token, context)
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
      @inbounds n_gram = join(enclosed_word[start:(start+s_i-1)])
      push!(n_grams, n_gram)
    end
  end
  return n_grams
end

"""
    create_dataset(documents::Vector{Vector{String}}, c::Int64, s_min::Int64, s_max::Int64, n::UInt32, max_unique_words::UInt64) -> Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}

Creates a dataset from a collection of documents.
Assumes the documents have already been preprocessed.
Creates context pairs with a context size of c,
subwords are constructed of ngrams with sizes between
s_min and s_max (inclusive). The hashing cutoff n
truncates the subword hashes. The max_unique_words
parameter ensures we can allocate the tracking data
structures beforehand, avoiding expensive resizing.
"""
function create_dataset(documents::Vector{Vector{String}}, c::Int64, s_min::Int64, s_max::Int64, n::UInt32)::Tuple{Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, Vector{UInt64}, Dict{String, UInt64}}
  labels = Dict{String, UInt64}()
  words = String[]
  counts = UInt64[]
  count = UInt64(1)
  context_pair_documents = Vector{Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}}(undef, length(documents))
  for (i, document) in enumerate(documents)
    labelled_document, count = assign_labels!(document, labels, words, counts, count)
    context_pair_document = pair_contexts(labelled_document, c)
    context_subword_document = Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}(undef, length(context_pair_document))
    for (j, (word, context)) in enumerate(context_pair_document)
      @inbounds subwords = AdaSubGram.Hashing.hash_words(split_subwords(words[word], s_min, s_max), n)
      @inbounds context_subword_document[j] = (word, subwords, context)
    end
    @inbounds context_pair_documents[i] = context_subword_document
  end
  return vcat(context_pair_documents...), counts, labels
end

"""
    minibatches(dataset::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, batch_size::Int64)::Vector{Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}}

Split up the dataset into minibatches
of size batch_size. Shuffles the
dataset first, to ensure it varies
throughout the training loop.
"""
function minibatches(dataset::Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}, batch_size::Int64)::Vector{Vector{Tuple{UInt64, Vector{UInt32}, Vector{UInt64}}}}
  shuffle!(dataset)
  return [dataset[i:min(i+batch_size-1, end)] for i in 1:batch_size:length(dataset)]
end

export create_dataset, minibatches, split_subwords

end
